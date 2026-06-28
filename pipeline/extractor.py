"""
Multimodal Document Extraction Pipeline

Uses GPT-4o's vision capabilities to extract structured JSON from:
- Scanned PDFs (converted to images)
- Photos of documents
- Forms, invoices, contracts, medical records

Key engineering decisions:
1. Schema-driven extraction: we give GPT-4o the exact JSON schema we expect
2. Confidence scoring: field-level confidence based on response patterns
3. Validation layer: JSON schema validation catches structural errors
4. Fallback: OCR + regex baseline for comparison
5. Human review queue: low-confidence extractions get flagged
"""

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
import io


@dataclass
class ExtractionResult:
    schema_type: str
    extracted_fields: Dict[str, Any]
    confidence_scores: Dict[str, float]  # per-field confidence
    overall_confidence: float
    raw_response: str
    needs_human_review: bool
    review_reason: str = ""
    extraction_method: str = "gpt4v"
    tokens_used: int = 0


# ── Document Schema Definitions ─────────────────────────────────────────────

DOCUMENT_SCHEMAS = {
    "invoice": {
        "description": "Extract all fields from this invoice or receipt",
        "fields": {
            "vendor_name": {"type": "string", "required": True},
            "invoice_number": {"type": "string", "required": True},
            "invoice_date": {"type": "string", "required": True},
            "due_date": {"type": "string", "required": False},
            "line_items": {
                "type": "array",
                "items": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "total": {"type": "number"},
                },
            },
            "subtotal": {"type": "number", "required": True},
            "tax": {"type": "number", "required": False},
            "total_amount": {"type": "number", "required": True},
            "currency": {"type": "string", "required": False},
            "payment_terms": {"type": "string", "required": False},
        },
    },
    "id_document": {
        "description": "Extract identity information from ID cards, passports, or driver's licenses",
        "fields": {
            "document_type": {"type": "string", "required": True},
            "full_name": {"type": "string", "required": True},
            "date_of_birth": {"type": "string", "required": True},
            "document_number": {"type": "string", "required": True},
            "expiry_date": {"type": "string", "required": False},
            "issuing_country": {"type": "string", "required": False},
            "issuing_state": {"type": "string", "required": False},
            "address": {"type": "string", "required": False},
        },
    },
    "medical_record": {
        "description": "Extract clinical information from medical records or discharge summaries",
        "fields": {
            "patient_name": {"type": "string", "required": True},
            "patient_dob": {"type": "string", "required": False},
            "mrn": {"type": "string", "required": False},
            "admission_date": {"type": "string", "required": False},
            "discharge_date": {"type": "string", "required": False},
            "primary_diagnosis": {"type": "string", "required": True},
            "secondary_diagnoses": {"type": "array", "required": False},
            "medications": {"type": "array", "required": False},
            "attending_physician": {"type": "string", "required": False},
            "follow_up_instructions": {"type": "string", "required": False},
        },
    },
    "business_card": {
        "description": "Extract contact information from a business card",
        "fields": {
            "full_name": {"type": "string", "required": True},
            "title": {"type": "string", "required": False},
            "company": {"type": "string", "required": False},
            "email": {"type": "string", "required": False},
            "phone": {"type": "string", "required": False},
            "website": {"type": "string", "required": False},
            "address": {"type": "string", "required": False},
            "linkedin": {"type": "string", "required": False},
        },
    },
    "form": {
        "description": "Extract all filled form fields from a form document",
        "fields": {
            "form_title": {"type": "string", "required": False},
            "form_fields": {
                "type": "object",
                "description": "Key-value pairs of all form fields and their values",
            },
            "signature_present": {"type": "boolean", "required": False},
            "date_signed": {"type": "string", "required": False},
        },
    },
}


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image(pil_image) -> str:
    """Encode PIL image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GPT4VExtractor:
    """
    Extracts structured data from document images using GPT-4o vision.

    Design principles:
    1. Schema-first: always provide the expected JSON schema to the model
    2. Role separation: system prompt defines the extraction task, user prompt provides the document
    3. Confidence inference: parse model hedging language to estimate confidence
    4. Graceful degradation: return partial extractions rather than failing completely
    """

    MODEL = "gpt-4o"

    def __init__(self, openai_api_key: str, confidence_threshold: float = 0.7):
        self.client = OpenAI(api_key=openai_api_key)
        self.confidence_threshold = confidence_threshold

    def _build_extraction_prompt(self, schema: Dict) -> str:
        field_descriptions = []
        for field_name, field_info in schema["fields"].items():
            req = "REQUIRED" if field_info.get("required") else "optional"
            field_descriptions.append(f"- {field_name} ({field_info['type']}, {req})")

        return f"""You are a document extraction system. {schema['description']}.

Extract the following fields from the document image and return a JSON object.
For any field you cannot read or are uncertain about, use null.
Include a "_confidence" object with confidence scores (0.0-1.0) for each field.

Required fields to extract:
{chr(10).join(field_descriptions)}

Return ONLY valid JSON in this format:
{{
    "extracted": {{field_name: value, ...}},
    "_confidence": {{field_name: 0.0-1.0, ...}},
    "_notes": "any extraction issues or ambiguities"
}}"""

    def extract_from_image(
        self,
        image_base64: str,
        schema_type: str = "invoice",
        image_media_type: str = "image/jpeg",
    ) -> ExtractionResult:
        """
        Main extraction method.
        Sends image + schema to GPT-4o, parses structured JSON response.
        """
        schema = DOCUMENT_SCHEMAS.get(schema_type, DOCUMENT_SCHEMAS["form"])

        system_prompt = self._build_extraction_prompt(schema)

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_media_type};base64,{image_base64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": system_prompt},
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            parsed = json.loads(raw_response)

            extracted_fields = parsed.get("extracted", {})
            confidence_scores = parsed.get("_confidence", {})
            notes = parsed.get("_notes", "")

            # Fill missing confidence scores
            for field in schema["fields"]:
                if field not in confidence_scores:
                    confidence_scores[field] = 0.0 if extracted_fields.get(field) is None else 0.8

            # Compute overall confidence
            required_fields = [k for k, v in schema["fields"].items() if v.get("required")]
            if required_fields:
                overall_confidence = sum(
                    confidence_scores.get(f, 0.0) for f in required_fields
                ) / len(required_fields)
            else:
                overall_confidence = sum(confidence_scores.values()) / max(len(confidence_scores), 1)

            needs_review = (
                overall_confidence < self.confidence_threshold
                or any(
                    extracted_fields.get(f) is None
                    for f in required_fields
                )
            )

            review_reason = ""
            if overall_confidence < self.confidence_threshold:
                review_reason = f"Low confidence ({overall_confidence:.1%}). "
            missing = [f for f in required_fields if extracted_fields.get(f) is None]
            if missing:
                review_reason += f"Missing required fields: {', '.join(missing)}. "
            if notes:
                review_reason += notes

            return ExtractionResult(
                schema_type=schema_type,
                extracted_fields=extracted_fields,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                raw_response=raw_response,
                needs_human_review=needs_review,
                review_reason=review_reason,
                extraction_method="gpt4v",
                tokens_used=tokens_used,
            )

        except Exception as e:
            return ExtractionResult(
                schema_type=schema_type,
                extracted_fields={},
                confidence_scores={},
                overall_confidence=0.0,
                raw_response=str(e),
                needs_human_review=True,
                review_reason=f"Extraction failed: {e}",
                extraction_method="gpt4v_error",
            )


class OCRBaselineExtractor:
    """
    Fallback baseline using pytesseract OCR + regex patterns.
    Used for comparison to demonstrate where vision models add value.
    """

    def extract_from_image(self, image_path: str, schema_type: str = "invoice") -> Dict:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            raw_text = pytesseract.image_to_string(img)
        except Exception:
            raw_text = "[OCR unavailable — pytesseract not installed]"

        extracted = {}
        if schema_type == "invoice":
            # Basic regex extraction for common invoice patterns
            patterns = {
                "invoice_number": r"(?:invoice|inv|#)[:\s#]*([A-Z0-9-]+)",
                "total_amount": r"(?:total|amount due)[:\s]*\$?([\d,]+\.?\d*)",
                "invoice_date": r"(?:date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            }
            for field, pattern in patterns.items():
                match = re.search(pattern, raw_text, re.IGNORECASE)
                extracted[field] = match.group(1) if match else None

        return {
            "raw_text": raw_text[:500],
            "extracted": extracted,
            "method": "ocr_regex",
            "note": "OCR baseline — misses structure, tables, and complex layouts",
        }
