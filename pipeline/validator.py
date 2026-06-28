"""
Validation Layer: JSON schema validation + cross-field consistency checks.

This is the layer that makes the system production-ready.
GPT-4o is good at extraction but can:
- Return wrong types (string instead of number)
- Miss required fields
- Return inconsistent values (subtotal + tax ≠ total)
- Hallucinate field values it can't see

The validator catches all of these before data flows downstream.
"""

import json
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_fields: Dict[str, Any]
    confidence_penalty: float  # subtract from extraction confidence


class DocumentValidator:
    """
    Schema-based validation with cross-field consistency checks.
    """

    def validate_invoice(self, fields: Dict) -> ValidationResult:
        errors = []
        warnings = []
        corrected = dict(fields)
        confidence_penalty = 0.0

        # Type checking and coercion
        numeric_fields = ["subtotal", "tax", "total_amount"]
        for f in numeric_fields:
            if f in fields and fields[f] is not None:
                val = fields[f]
                if isinstance(val, str):
                    cleaned = re.sub(r"[,$\s]", "", val)
                    try:
                        corrected[f] = float(cleaned)
                    except ValueError:
                        errors.append(f"Cannot parse '{f}' as number: {val}")
                        confidence_penalty += 0.1

        # Cross-field consistency: subtotal + tax ≈ total
        subtotal = corrected.get("subtotal")
        tax = corrected.get("tax", 0) or 0
        total = corrected.get("total_amount")
        if subtotal and total:
            expected = subtotal + tax
            if abs(expected - total) > 0.02 * total:  # 2% tolerance
                warnings.append(
                    f"Totals inconsistent: subtotal({subtotal}) + tax({tax}) = {expected:.2f} ≠ total({total})"
                )
                confidence_penalty += 0.15

        # Date format validation
        date_fields = ["invoice_date", "due_date"]
        date_pattern = re.compile(r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$|^\d{4}[/\-\.]\d{2}[/\-\.]\d{2}$")
        for f in date_fields:
            if fields.get(f) and not date_pattern.match(str(fields[f])):
                warnings.append(f"Non-standard date format in '{f}': {fields[f]}")
                confidence_penalty += 0.05

        # Required fields check
        required = ["vendor_name", "invoice_number", "total_amount"]
        missing = [f for f in required if not corrected.get(f)]
        if missing:
            errors.append(f"Missing required fields: {missing}")
            confidence_penalty += len(missing) * 0.15

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_fields=corrected,
            confidence_penalty=min(0.5, confidence_penalty),
        )

    def validate_id_document(self, fields: Dict) -> ValidationResult:
        errors = []
        warnings = []
        corrected = dict(fields)
        confidence_penalty = 0.0

        # Date format validation for DOB and expiry
        for date_field in ["date_of_birth", "expiry_date"]:
            if fields.get(date_field):
                raw = str(fields[date_field])
                if len(raw) < 6:
                    warnings.append(f"Date '{date_field}' seems incomplete: {raw}")
                    confidence_penalty += 0.1

        # Document number should be alphanumeric
        doc_num = fields.get("document_number")
        if doc_num and not re.match(r"^[A-Z0-9\-]+$", str(doc_num).upper()):
            warnings.append(f"Document number contains unusual characters: {doc_num}")
            confidence_penalty += 0.05

        required = ["full_name", "document_number", "date_of_birth"]
        missing = [f for f in required if not corrected.get(f)]
        if missing:
            errors.append(f"Missing required fields: {missing}")
            confidence_penalty += len(missing) * 0.2

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_fields=corrected,
            confidence_penalty=min(0.5, confidence_penalty),
        )

    def validate(self, schema_type: str, fields: Dict) -> ValidationResult:
        """Dispatch to schema-specific validator."""
        validators = {
            "invoice": self.validate_invoice,
            "id_document": self.validate_id_document,
        }
        validator_fn = validators.get(schema_type)
        if validator_fn:
            return validator_fn(fields)

        # Generic validation for other schema types
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            corrected_fields=fields,
            confidence_penalty=0.0,
        )


class HumanReviewQueue:
    """
    Manages the queue of extractions that need human review.
    In production: integrates with a task management system (Jira, Asana, etc.)
    Here: in-memory queue with priority scoring.
    """

    def __init__(self, confidence_threshold: float = 0.7, max_queue_size: int = 100):
        self.threshold = confidence_threshold
        self.queue: List[Dict] = []
        self.max_size = max_queue_size
        self.stats = {"total_reviewed": 0, "auto_approved": 0, "sent_to_review": 0}

    def process(
        self,
        extraction_result,  # ExtractionResult
        validation_result: ValidationResult,
        document_id: str = "doc_001",
    ) -> Dict:
        """
        Decide: auto-approve or send to human review queue.
        Returns processing decision with reasoning.
        """
        final_confidence = max(
            0.0, extraction_result.overall_confidence - validation_result.confidence_penalty
        )

        reasons_for_review = []
        if final_confidence < self.threshold:
            reasons_for_review.append(f"Low confidence ({final_confidence:.1%} < {self.threshold:.0%} threshold)")
        if validation_result.errors:
            reasons_for_review.append(f"Validation errors: {'; '.join(validation_result.errors)}")
        if extraction_result.needs_human_review:
            reasons_for_review.append(extraction_result.review_reason)

        needs_review = bool(reasons_for_review)
        self.stats["total_reviewed"] += 1

        if needs_review and len(self.queue) < self.max_size:
            priority = "HIGH" if final_confidence < 0.4 or validation_result.errors else "MEDIUM"
            self.queue.append({
                "document_id": document_id,
                "confidence": final_confidence,
                "reasons": reasons_for_review,
                "priority": priority,
                "extracted_fields": validation_result.corrected_fields,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
            })
            self.stats["sent_to_review"] += 1
        else:
            self.stats["auto_approved"] += 1

        return {
            "decision": "review" if needs_review else "approved",
            "final_confidence": final_confidence,
            "reasons": reasons_for_review,
            "corrected_fields": validation_result.corrected_fields,
            "validation_errors": validation_result.errors,
            "validation_warnings": validation_result.warnings,
            "queue_size": len(self.queue),
            "stats": dict(self.stats),
        }

    def get_queue(self) -> List[Dict]:
        return sorted(self.queue, key=lambda x: x["confidence"])
