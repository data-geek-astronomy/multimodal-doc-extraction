---
title: Multimodal Document Extraction Pipeline
emoji: 📄
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: GPT-4o structured JSON extraction with validation layer
python_version: "3.10"
---

# 📄 Multimodal Document Extraction Pipeline

> GPT-4o vision extracts structured JSON from any document image. Schema-based validation + cross-field consistency checks + automatic human review routing for low-confidence outputs.

## Architecture

```
Document Image
    ↓
GPT-4o Vision Extraction (schema-driven, json_object mode, per-field confidence)
    ↓
JSON Schema Validation (type coercion, cross-field consistency, required fields)
    ↓
Confidence Scoring (extraction confidence - validation penalty)
    ↓
Human Review Queue (auto-approve if ≥70% AND no errors, else queue)
    ↓
Structured JSON Output
```

## Supported Document Types

| Type | Key Fields |
|---|---|
| Invoice | Vendor, invoice #, line items, totals, tax, payment terms |
| ID Document | Name, DOB, document #, expiry, issuing authority |
| Medical Record | Patient info, diagnoses, medications, follow-up |
| Business Card | Name, title, company, email, phone, website |
| Form | All form fields as key-value pairs, signature |

## Key Engineering Decisions

**Schema-first extraction**: Providing GPT-4o the exact expected JSON schema reduces hallucination and forces consistent field naming. The model returns `null` for fields it cannot read (vs. omitting them), which distinguishes "field not found" from "field not present."

**Per-field confidence**: The extraction prompt explicitly requests confidence scores (0-1) per field. These are used downstream for validation penalty calculation.

**Cross-field consistency**: Invoice totals are cross-validated: `subtotal + tax ≈ total_amount` within 2% tolerance. Violations trigger a confidence penalty and human review.

**Validation penalty**: `final_confidence = extraction_confidence - validation_penalty` where penalty accumulates per error type (type mismatch: -0.10, consistency error: -0.15, missing required: -0.15).

## Vision vs OCR Baseline

| Capability | GPT-4o | Tesseract + Regex |
|---|---|---|
| Tables / line items | ✅ Full structure | ❌ Loses structure |
| Handwritten text | ✅ Good | ❌ Very poor |
| Multi-language | ✅ Built-in | ⚠️ Needs config |
| Speed | ~3 sec | ~0.3 sec |
| Cost | ~$0.02/doc | Free |

## Running Locally

```bash
git clone https://github.com/data-geek-astronomy/multimodal-doc-extraction
cd multimodal-doc-extraction
pip install -r requirements.txt
OPENAI_API_KEY=sk-... python app.py
```

## License

MIT
