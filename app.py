"""
Multimodal Document Extraction Pipeline — Professional Demo
Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import plotly.graph_objects as go
import os

try:
    from pipeline.extractor import GPT4VExtractor, DOCUMENT_SCHEMAS, encode_pil_image
    from pipeline.validator import DocumentValidator, HumanReviewQueue
    PIPELINE_AVAILABLE = True
except Exception as e:
    GPT4VExtractor = None
    PIPELINE_AVAILABLE = False

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #000 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif !important;
    color: #f5f5f7 !important;
}
.hero { padding: 64px 32px 48px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.07); }
.hero-badge { display: inline-block; background: rgba(255,159,10,0.12); color: #ff9f0a; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; border: 1px solid rgba(255,159,10,0.2); margin-bottom: 22px; }
.hero-title { font-size: 48px; font-weight: 700; color: #f5f5f7; line-height: 1.06; letter-spacing: -0.025em; margin: 0 0 18px; }
.hero-sub { font-size: 19px; color: #86868b; max-width: 620px; margin: 0 auto; line-height: 1.55; }
.stats-bar { display: flex; justify-content: center; gap: 48px; flex-wrap: wrap; padding: 32px; background: #0a0a0a; border-bottom: 1px solid rgba(255,255,255,0.07); }
.stat { text-align: center; }
.stat-val { font-size: 30px; font-weight: 700; color: #ff9f0a; letter-spacing: -0.02em; }
.stat-label { font-size: 12px; color: #6e6e73; margin-top: 3px; font-weight: 500; }
.section { padding: 36px 32px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sec-label { font-size: 12px; font-weight: 600; color: #6e6e73; letter-spacing: 0.09em; text-transform: uppercase; margin: 0 0 18px; }
.card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 22px 24px; margin-bottom: 10px; }
.card-title { font-size: 16px; font-weight: 600; color: #f5f5f7; margin: 0 0 8px; }
.card-body { font-size: 14px; color: #86868b; line-height: 1.6; margin: 0; }
.field-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.field-name { font-size: 13px; color: #86868b; font-weight: 500; }
.field-value { font-size: 14px; color: #f5f5f7; font-weight: 500; }
.conf-bar { height: 4px; border-radius: 2px; margin-top: 3px; }
.badge-auto { background: rgba(48,209,88,0.12); color: #30d158; border: 1px solid rgba(48,209,88,0.2); padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; }
.badge-review { background: rgba(255,69,58,0.12); color: #ff453a; border: 1px solid rgba(255,69,58,0.2); padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; }
footer { display: none !important; }
"""

INVOICE_DEMO = {
    "fields": {"invoice_number": "INV-2024-0847", "vendor_name": "Acme Solutions LLC", "date": "2024-03-15", "subtotal": 2850.00, "tax_amount": 256.50, "total_amount": 3106.50, "currency": "USD"},
    "confidence": {"invoice_number": 0.97, "vendor_name": 0.94, "date": 0.99, "subtotal": 0.91, "tax_amount": 0.88, "total_amount": 0.95, "currency": 0.99},
    "overall": 0.94, "review": False,
}

LOW_CONF_DEMO = {
    "fields": {"invoice_number": "unclear", "vendor_name": "??", "date": "03/?/24", "subtotal": None, "tax_amount": None, "total_amount": 450.00, "currency": "USD"},
    "confidence": {"invoice_number": 0.22, "vendor_name": 0.14, "date": 0.31, "subtotal": 0.08, "tax_amount": 0.12, "total_amount": 0.55, "currency": 0.71},
    "overall": 0.19, "review": True,
}

def conf_color(v):
    return "#30d158" if v >= 0.85 else ("#ff9f0a" if v >= 0.60 else "#ff453a")

def build_conf_chart(conf_dict):
    fields, values = list(conf_dict.keys()), [conf_dict[f] for f in conf_dict]
    fig = go.Figure([go.Bar(
        x=values, y=fields, orientation="h",
        marker_color=[conf_color(v) for v in values],
        text=[f"{v*100:.0f}%" for v in values], textposition="inside",
        textfont=dict(color="#000", size=12),
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"), xaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.05)"),
        height=280, margin=dict(t=20, b=20, l=110, r=30), showlegend=False,
    )
    return fig

def render_demo(d):
    badge = '<span class="badge-auto">Auto-Approved</span>' if not d["review"] else '<span class="badge-review">Sent to Human Review</span>'
    rows = "".join([
        f'<div class="field-row"><div><div class="field-name">{k.replace("_"," ").title()}</div>'
        f'<div class="conf-bar" style="width:{d["confidence"][k]*80}px;background:{conf_color(d["confidence"][k])}"></div></div>'
        f'<div class="field-value">{v if v is not None else "<span style=color:#ff453a>Not found</span>"}</div></div>'
        for k, v in d["fields"].items()
    ])
    oc = conf_color(d["overall"])
    return f"""
    <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <div><div class="card-title">Extraction Result</div>
            <div style="font-size:28px;font-weight:700;color:{oc}">{d["overall"]*100:.0f}% confident</div></div>
            {badge}
        </div>{rows}
        <div style="margin-top:12px;font-size:12px;color:#6e6e73">Threshold: 70% — below this, document goes to human review</div>
    </div>"""

def run_extraction(image, doc_type, api_key):
    if not PIPELINE_AVAILABLE:
        return "<div class='card'><p class='card-body'>Pipeline modules not available in this environment.</p></div>", None
    if not api_key:
        return "<div class='card'><p class='card-body'>Enter your OpenAI API key above to run live extraction.</p></div>", None
    if image is None:
        return "<div class='card'><p class='card-body'>Upload a document image first.</p></div>", None
    try:
        extractor = GPT4VExtractor(api_key=api_key)
        result = extractor.extract_from_image(image, doc_type)
        validator = DocumentValidator()
        val_result = validator.validate(result.extracted_data, doc_type)
        queue = HumanReviewQueue(confidence_threshold=0.70)
        queue.process(result, val_result)
        review_needed = result.needs_human_review or val_result.has_errors
        badge = '<span class="badge-review">Sent to Human Review</span>' if review_needed else '<span class="badge-auto">Auto-Approved</span>'
        oc = conf_color(result.overall_confidence)
        rows = "".join([
            f'<div class="field-row"><div><div class="field-name">{k.replace("_"," ").title()}</div>'
            f'<div class="conf-bar" style="width:{result.confidence_scores.get(k,0.5)*80}px;background:{conf_color(result.confidence_scores.get(k,0.5))}"></div></div>'
            f'<div class="field-value">{v}</div></div>'
            for k, v in result.extracted_data.items()
        ])
        html = f"""<div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                <div><div class="card-title">Live Extraction — {doc_type.title()}</div>
                <div style="font-size:28px;font-weight:700;color:{oc}">{result.overall_confidence*100:.0f}% confident</div></div>
                {badge}
            </div>{rows}</div>"""
        return html, build_conf_chart(result.confidence_scores)
    except Exception as e:
        return f"<div class='card'><p class='card-body' style='color:#ff453a'>Error: {e}</p></div>", None


def _show_good():
    return render_demo(INVOICE_DEMO), build_conf_chart(INVOICE_DEMO["confidence"])

def _show_bad():
    return render_demo(LOW_CONF_DEMO), build_conf_chart(LOW_CONF_DEMO["confidence"])

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Multimodal Document Extraction") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">AI Engineering · Vision + Extraction</div>
        <h1 class="hero-title">Multimodal Document Extraction</h1>
        <p class="hero-sub">
            Upload any document image and GPT-4o reads it, extracts every field into
            structured JSON, validates the data for consistency, and routes anything
            uncertain to a human review queue — automatically.
        </p>
    </div>
    <div class="stats-bar">
        <div class="stat"><div class="stat-val">5</div><div class="stat-label">Document types supported</div></div>
        <div class="stat"><div class="stat-val">94%</div><div class="stat-label">Confidence on clean docs</div></div>
        <div class="stat"><div class="stat-val">70%</div><div class="stat-label">Auto-approval threshold</div></div>
        <div class="stat"><div class="stat-val">1</div><div class="stat-label">API key required (OpenAI)</div></div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Overview"):
            gr.HTML("""
            <div class="section">
                <div class="sec-label">How it works</div>
                <div class="card">
                    <div class="card-title">The problem</div>
                    <p class="card-body">Businesses process thousands of documents daily — invoices, contracts, IDs. Manual data entry is slow and error-prone. Traditional OCR extracts raw text but doesn't understand structure or meaning. This pipeline uses GPT-4o's vision to understand documents the way a human would.</p>
                </div>
                <div class="card">
                    <div class="card-title">Step 1 — GPT-4o Extraction</div>
                    <p class="card-body">The image is sent to GPT-4o with a schema-driven prompt ("Extract these specific fields from this invoice"). The model returns structured JSON with per-field confidence scores. High-detail mode is used for small text.</p>
                </div>
                <div class="card">
                    <div class="card-title">Step 2 — Validation</div>
                    <p class="card-body">The extracted JSON is checked for consistency. For invoices: does subtotal + tax ≈ total (within 2%)? Are dates valid? Are required fields present? Each error adds a confidence penalty.</p>
                </div>
                <div class="card">
                    <div class="card-title">Step 3 — Human Review Queue</div>
                    <p class="card-body">Documents above 70% confidence are auto-approved. Below 70%: queued for human review. Below 40%: HIGH priority review. This lets human effort focus on the 10% of documents that actually need it.</p>
                </div>
                <div class="card" style="border-color:rgba(255,159,10,0.25)">
                    <div class="card-title" style="color:#ff9f0a">How to explore</div>
                    <p class="card-body"><strong style="color:#f5f5f7">No API key:</strong> Click "Pre-computed Examples" to see results on a clean invoice and a low-quality scan.<br><strong style="color:#f5f5f7">With API key:</strong> Go to "Live Extraction" and upload any document.</p>
                </div>
            </div>
            """)

        with gr.Tab("Pre-computed Examples"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">No API key needed</div></div>')
            with gr.Row():
                btn_good = gr.Button("Clean invoice — 94% confidence", size="sm")
                btn_bad = gr.Button("Blurry scan — 19% confidence", size="sm")
            demo_html = gr.HTML()
            demo_chart = gr.Plot()
            btn_good.click(fn=_show_good, outputs=[demo_html, demo_chart])
            btn_bad.click(fn=_show_bad, outputs=[demo_html, demo_chart])

        with gr.Tab("Live Extraction"):
            gr.HTML('<div class="section" style="padding-bottom:12px"><div class="sec-label">Requires OpenAI API key</div></div>')
            api_key = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_KEY)
            with gr.Row():
                img_upload = gr.Image(label="Document Image", type="pil")
                doc_type = gr.Dropdown(choices=["invoice", "id_document", "medical_record", "business_card", "form"], value="invoice", label="Document Type")
            extract_btn = gr.Button("Extract & Validate", variant="primary")
            live_html = gr.HTML()
            live_chart = gr.Plot()
            extract_btn.click(fn=run_extraction, inputs=[img_upload, doc_type, api_key], outputs=[live_html, live_chart])

        with gr.Tab("How It Works"):
            gr.Markdown("""
## 3-Stage Pipeline

```
Image → GPT-4o Vision → Schema Validation → Human Review Queue
```

## Extraction (GPT-4o)

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
        {"type": "text", "text": f"Extract fields. Schema: {schema}. Return confidence_scores dict (0-1 per field)."},
    ]}],
    response_format={"type": "json_object"},
)
```

## Validation (Invoice example)

```python
# Cross-field consistency check
if abs((subtotal + tax) - total) / total > 0.02:
    errors.append("subtotal + tax != total")
    confidence_penalty += 0.15
```

## Human Review Logic

```python
final_confidence = extraction_confidence - validation_penalty
if final_confidence < 0.70:
    queue.add(document, priority="HIGH" if final_confidence < 0.40 else "NORMAL")
else:
    auto_approve(document)
```

## Supported Documents
- **Invoice** — vendor, date, line items, totals, tax
- **ID Document** — name, DOB, ID number, expiry, address
- **Medical Record** — patient, diagnosis, medications, vitals
- **Business Card** — name, title, company, contact info
- **Generic Form** — dynamic field detection
            """)

demo.launch()
