"""
Multimodal Document Extraction Pipeline — Interactive Demo
==========================================================
Upload any document image → GPT-4o extracts structured JSON →
Validator checks consistency → Human review queue for low-confidence outputs.

Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import json
import os
from PIL import Image
import io
import base64
import plotly.graph_objects as go

from pipeline import (
    GPT4VExtractor, DocumentValidator, HumanReviewQueue,
    DOCUMENT_SCHEMAS, encode_pil_image,
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

review_queue = HumanReviewQueue(confidence_threshold=0.70)
extractor = None
validator = DocumentValidator()

CSS = """
body, .gradio-container { background: #0a0d14 !important; }
.card { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 18px; margin: 8px 0; }
.high-conf { border-color: #22c55e !important; }
.low-conf { border-color: #ef4444 !important; }
footer { display: none !important; }
"""

DEMO_RESULTS = {
    "invoice_demo": {
        "extracted": {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-2024-0847",
            "invoice_date": "2024-03-15",
            "due_date": "2024-04-15",
            "line_items": [
                {"description": "Software License (Annual)", "quantity": 1, "unit_price": 2400.00, "total": 2400.00},
                {"description": "Implementation Services", "quantity": 40, "unit_price": 150.00, "total": 6000.00},
                {"description": "Training (2 sessions)", "quantity": 2, "unit_price": 500.00, "total": 1000.00},
            ],
            "subtotal": 9400.00,
            "tax": 846.00,
            "total_amount": 10246.00,
            "currency": "USD",
            "payment_terms": "Net 30",
        },
        "confidence": {"vendor_name": 0.98, "invoice_number": 0.97, "total_amount": 0.95, "line_items": 0.89, "tax": 0.92},
        "overall_confidence": 0.94,
        "decision": "approved",
        "errors": [],
        "warnings": [],
    },
    "low_confidence_demo": {
        "extracted": {
            "vendor_name": None,
            "invoice_number": "INV-???",
            "invoice_date": "unclear",
            "total_amount": None,
        },
        "confidence": {"vendor_name": 0.0, "invoice_number": 0.45, "invoice_date": 0.3, "total_amount": 0.0},
        "overall_confidence": 0.19,
        "decision": "review",
        "errors": ["Missing required fields: ['vendor_name', 'total_amount']"],
        "warnings": ["Non-standard date format in 'invoice_date': unclear"],
    },
}


def run_extraction(image, schema_type: str, api_key: str):
    if image is None:
        return (
            "<div class='card'>📤 Upload a document image to begin extraction.</div>",
            "{}", "<div class='card'>Awaiting image...</div>", None
        )

    if not api_key:
        return (
            "<div class='card'>❌ Enter your OpenAI API key.</div>",
            "{}", "", None
        )

    try:
        pil_img = Image.fromarray(image).convert("RGB")
        img_b64 = encode_pil_image(pil_img)
    except Exception as e:
        return f"<div class='card'>❌ Image error: {e}</div>", "{}", "", None

    global extractor
    if extractor is None or extractor.client.api_key != api_key:
        extractor = GPT4VExtractor(api_key, confidence_threshold=0.70)

    result = extractor.extract_from_image(img_b64, schema_type=schema_type)
    val_result = validator.validate(schema_type, result.extracted_fields)
    decision = review_queue.process(result, val_result)

    # Build result HTML
    conf = decision["final_confidence"]
    conf_color = "#22c55e" if conf >= 0.70 else "#f59e0b" if conf >= 0.50 else "#ef4444"
    conf_label = "✅ AUTO-APPROVED" if decision["decision"] == "approved" else "⚠️ SENT TO REVIEW"

    fields_html = ""
    for k, v in decision["corrected_fields"].items():
        if v is not None and k != "_confidence" and k != "_notes":
            field_conf = result.confidence_scores.get(k, 0.0)
            fc = "#22c55e" if field_conf >= 0.8 else "#f59e0b" if field_conf >= 0.5 else "#ef4444"
            val_str = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
            fields_html += f"""
            <div style='display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                <div style='color:{fc};font-size:0.75em;min-width:40px;margin-top:2px'>{field_conf:.0%}</div>
                <div style='flex:1'>
                    <div style='color:#64748b;font-size:0.75em'>{k}</div>
                    <div style='color:#e2e8f0;font-size:0.88em;font-family:monospace;word-break:break-all'>{val_str[:200]}</div>
                </div>
            </div>"""

    errors_html = ""
    if decision["validation_errors"]:
        errors_html = f"<div style='color:#ef4444;font-size:0.82em;margin-top:8px'>❌ {'; '.join(decision['validation_errors'])}</div>"
    if decision["validation_warnings"]:
        errors_html += f"<div style='color:#f59e0b;font-size:0.82em;margin-top:4px'>⚠️ {'; '.join(decision['validation_warnings'])}</div>"

    result_html = f"""
    <div class='card {"high-conf" if decision["decision"]=="approved" else "low-conf"}'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'>
            <div>
                <div style='color:{conf_color};font-weight:700;font-size:1.05em'>{conf_label}</div>
                <div style='color:#64748b;font-size:0.8em'>{schema_type} · {result.tokens_used} tokens used</div>
            </div>
            <div style='text-align:right'>
                <div style='color:{conf_color};font-size:2em;font-weight:700'>{conf:.0%}</div>
                <div style='color:#64748b;font-size:0.75em'>Overall Confidence</div>
            </div>
        </div>
        {fields_html}
        {errors_html}
        <div style='color:#475569;font-size:0.78em;margin-top:10px'>
            Review queue: {decision["queue_size"]} items |
            Auto-approved: {decision["stats"]["auto_approved"]} |
            Sent to review: {decision["stats"]["sent_to_review"]}
        </div>
    </div>
    """

    json_output = json.dumps(decision["corrected_fields"], indent=2)
    queue_html = build_queue_html()
    conf_chart = build_confidence_chart(result.confidence_scores)

    return result_html, json_output, queue_html, conf_chart


def build_confidence_chart(confidence_scores: dict):
    if not confidence_scores:
        return None
    fields = list(confidence_scores.keys())
    scores = [confidence_scores[f] for f in fields]
    colors = ["#22c55e" if s >= 0.8 else "#f59e0b" if s >= 0.5 else "#ef4444" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=fields, orientation="h",
        marker_color=colors,
        text=[f"{s:.0%}" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Per-Field Confidence Scores",
        xaxis=dict(range=[0, 1.1]),
        height=max(200, len(fields) * 35 + 80),
        margin=dict(t=40, b=10, l=150, r=60),
    )
    return fig


def build_queue_html():
    queue = review_queue.get_queue()
    if not queue:
        return "<div class='card' style='color:#64748b'>No items in review queue.</div>"

    items_html = "".join(
        f"""<div style='background:rgba({"239,68,68" if item["priority"]=="HIGH" else "245,158,11"},0.07);border:1px solid {"#ef4444" if item["priority"]=="HIGH" else "#f59e0b"};border-radius:8px;padding:12px;margin:6px 0'>
            <div style='display:flex;justify-content:space-between'>
                <div style='color:#e2e8f0;font-size:0.9em;font-weight:600'>{item["document_id"]}</div>
                <div style='color:{"#ef4444" if item["priority"]=="HIGH" else "#f59e0b"};font-size:0.78em;font-weight:600'>{item["priority"]} PRIORITY</div>
            </div>
            <div style='color:#94a3b8;font-size:0.8em;margin-top:4px'>Confidence: {item["confidence"]:.1%}</div>
            <div style='color:#64748b;font-size:0.78em;margin-top:4px'>{'<br/>'.join(item["reasons"][:2])}</div>
        </div>"""
        for item in queue[:5]
    )
    return f"""<div class='card'><h4 style='color:#a5b4fc;margin:0 0 10px'>Review Queue ({len(queue)} items)</h4>{items_html}</div>"""


def load_demo_result(demo_type: str):
    d = DEMO_RESULTS.get(demo_type, DEMO_RESULTS["invoice_demo"])
    conf = d["overall_confidence"]
    conf_color = "#22c55e" if conf >= 0.70 else "#ef4444"
    label = "✅ AUTO-APPROVED" if d["decision"] == "approved" else "⚠️ SENT TO REVIEW"

    fields_html = "".join(
        f"<div style='padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
        f"<span style='color:#64748b;font-size:0.8em'>{k}: </span>"
        f"<span style='color:#e2e8f0;font-size:0.85em'>{json.dumps(v) if isinstance(v, list) else v}</span>"
        f"</div>"
        for k, v in d["extracted"].items() if v is not None
    )

    result_html = f"""
    <div class='card {"high-conf" if d["decision"]=="approved" else "low-conf"}'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
            <div style='color:{conf_color};font-weight:700'>{label}</div>
            <div style='color:{conf_color};font-size:2em;font-weight:700'>{conf:.0%}</div>
        </div>
        {fields_html}
        {''.join(f"<div style='color:#ef4444;font-size:0.82em;margin-top:6px'>❌ {e}</div>" for e in d.get("errors", []))}
        {''.join(f"<div style='color:#f59e0b;font-size:0.82em'>⚠️ {w}</div>" for w in d.get("warnings", []))}
    </div>"""

    conf_chart = build_confidence_chart(d["confidence"])
    return result_html, json.dumps(d["extracted"], indent=2), conf_chart


with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="violet"), title="Multimodal Doc Extraction") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:28px 0 18px'>
        <div style='font-size:2.8em'>📄</div>
        <h1 style='color:#e2e8f0;margin:10px 0 6px;font-size:1.9em;font-weight:700'>
            Multimodal Document Extraction
        </h1>
        <p style='color:#64748b;max-width:680px;margin:0 auto;line-height:1.6'>
            GPT-4o vision extracts structured JSON from any document image.
            Schema-based validation catches inconsistencies. Low-confidence
            extractions are automatically routed to human review.
        </p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("📤 Extract from Document"):
            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_KEY, scale=3)
                schema_select = gr.Dropdown(
                    choices=list(DOCUMENT_SCHEMAS.keys()),
                    value="invoice", label="Document Type", scale=1,
                )

            with gr.Row():
                image_input = gr.Image(label="Upload Document / Photo", type="numpy", scale=2)
                with gr.Column(scale=3):
                    result_display = gr.HTML(value="<div class='card'>Upload an image to begin.</div>")
                    conf_chart = gr.Plot()

            extract_btn = gr.Button("🔍 Extract Structured Data", variant="primary", size="lg")
            json_output = gr.Code(label="Extracted JSON", language="json", lines=12)
            queue_display = gr.HTML()

            extract_btn.click(
                fn=run_extraction,
                inputs=[image_input, schema_select, api_key],
                outputs=[result_display, json_output, queue_display, conf_chart],
            )

        with gr.Tab("🎯 Demo Results"):
            gr.HTML("""
            <div class='card'>
                <p style='color:#94a3b8;margin:0'>Pre-computed extraction results demonstrating the pipeline's output at high and low confidence levels.</p>
            </div>
            """)
            with gr.Row():
                demo1_btn = gr.Button("📋 High-Confidence Invoice (94%)", size="lg")
                demo2_btn = gr.Button("⚠️ Low-Confidence / Review Queue (19%)", size="lg")

            demo_result = gr.HTML()
            demo_json = gr.Code(label="Extracted JSON", language="json", lines=12)
            demo_chart = gr.Plot()

            demo1_btn.click(lambda: load_demo_result("invoice_demo"), outputs=[demo_result, demo_json, demo_chart])
            demo2_btn.click(lambda: load_demo_result("low_confidence_demo"), outputs=[demo_result, demo_json, demo_chart])

        with gr.Tab("🏗️ Architecture"):
            gr.Markdown("""
## Pipeline Design

```
Document Image
      ↓
GPT-4o Vision Extraction
  - Schema-driven prompt (tells GPT exactly what fields to extract)
  - response_format: json_object (guaranteed valid JSON)
  - high detail mode (better OCR from vision model)
  - Per-field confidence scores requested in prompt
      ↓
JSON Schema Validation
  - Type coercion (strings → numbers where expected)
  - Cross-field consistency (subtotal + tax = total?)
  - Required field checks
  - Date format validation
      ↓
Confidence Scoring
  - Final confidence = extraction confidence - validation penalty
  - Penalty: 0.1 per type error, 0.15 per consistency issue, 0.15 per missing required
      ↓
Human Review Queue Decision
  - confidence < 0.70 → queue (priority: HIGH if < 0.40)
  - validation errors → queue
  - confidence ≥ 0.70 AND no errors → auto-approve
      ↓
Structured JSON Output
```

## Why Schema-First Extraction?

Giving GPT-4o the exact expected JSON schema in the prompt:
1. **Reduces hallucination**: model knows what fields to look for, doesn't invent structure
2. **Improves field naming**: consistent snake_case keys regardless of document layout
3. **Forces null vs omission**: model returns null for invisible fields (vs omitting them)
4. **Enables per-field confidence**: model can rate its own uncertainty per field

## Vision vs OCR Baseline

| Capability | GPT-4o Vision | Tesseract OCR + Regex |
|---|---|---|
| Scanned forms | ✅ Excellent | ⚠️ Depends on image quality |
| Tables & line items | ✅ Full structure | ❌ Loses structure |
| Handwritten text | ✅ Good | ❌ Very poor |
| Low-res images | ⚠️ Degrades | ❌ Fails |
| Multi-language | ✅ Built-in | ⚠️ Needs language config |
| Speed | ~2-4 seconds | ~0.3 seconds |
| Cost | ~$0.01-0.05/doc | Free |

**Use vision models when**: structure matters (tables, forms, mixed layouts)
**Use OCR when**: simple text extraction from high-quality scans, cost-sensitive at scale
            """)

demo.launch()
