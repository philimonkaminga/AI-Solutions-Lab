import streamlit as st
import openai
import base64
import json
import pandas as pd
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Invoice Expert", page_icon="🧾", layout="centered")
st.title("🧾 Invoice Extractor Pro")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# OpenAI API Key
openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# File uploader with supported formats
uploaded_file = st.file_uploader("📤 Upload Invoice (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])

# Invoice schema validation
REQUIRED_KEYS = {
    "invoice_number": "string",
    "date": "string",
    "vendor": "string",
    "items": "array",
    "subtotal": "number",
    "tax": "number",
    "total": "number"
}

ITEM_KEYS = ["description", "quantity", "unit_price", "total_price"]

def validate_json(data):
    """Validate invoice JSON structure"""
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid invoice data format")
    
    missing_keys = [key for key in REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Missing fields: {', '.join(missing_keys)}")
    
    if not isinstance(data['items'], list) or len(data['items']) == 0:
        raise ValueError("Invalid items list")
    
    for item in data['items']:
        missing = [k for k in ITEM_KEYS if k not in item]
        if missing:
            raise ValueError(f"Item missing: {', '.join(missing)}")
        
        if not all(isinstance(item[k], (int, float)) for k in ['quantity', 'unit_price', 'total_price']):
            raise ValueError("Invalid numeric values in items")

def process_invoice(image_bytes):
    """Process invoice image using GPT-4 Vision"""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    messages = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }]
    
    messages.append({
        "type": "text",
        "text": """Extract structured invoice data as JSON with:
        - invoice_number (string)
        - date (string)
        - vendor (string)
        - items (array of {description, quantity, unit_price, total_price})
        - subtotal (number)
        - tax (number)
        - total (number)
        Return ONLY valid JSON without comments"""
    })

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": messages}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def pdf_to_jpeg(pdf_bytes, zoom=2.0):
    """Convert PDF to JPEG with adjustable zoom"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save as JPEG with quality preservation
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"PDF conversion failed: {str(e)}")
        st.stop()

def display_invoice(data):
    """Display extracted invoice data"""
    st.session_state.processed_data = data
    
    st.success("✅ Invoice data extracted successfully!")
    
    with st.container():
        cols = st.columns(3)
        cols[0].metric("Invoice Number", data.get('invoice_number', 'N/A'))
        cols[1].metric("Date", data.get('date', 'N/A'))
        cols[2].metric("Vendor", data.get('vendor', 'N/A'))

    st.subheader("Items List")
    items_df = pd.DataFrame(data['items'])
    items_df.index += 1
    st.dataframe(
        items_df.style.format({
            'unit_price': 'ZMW{:.2f}',
            'total_price': 'ZMW{:.2f}',
            'quantity': '{:.0f}'
        }),
        use_container_width=True,
        height=400
    )

    with st.container():
        cols = st.columns(3)
        cols[0].metric("Subtotal", f"ZMW{data['subtotal']:.2f}")
        cols[1].metric("Tax", f"ZMW{data['tax']:.2f}")
        cols[2].metric("Total", f"ZMW{data['total']:.2f}")

if uploaded_file and openai.api_key:
    try:
        image_bytes = None
        preview_width = 700  # Fixed preview width for better control
        
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, width=preview_width)
            image_bytes = uploaded_file.getvalue()
            
        elif uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            # Add zoom control for PDF previews
            zoom_level = st.slider("PDF Preview Zoom", 1.0, 3.0, 1.5, 0.1)
            image_bytes = pdf_to_jpeg(pdf_bytes, zoom=zoom_level)
            st.image(image_bytes, width=preview_width)

        if image_bytes and st.button("Extract Invoice Data"):
            with st.spinner("Analyzing document..."):
                result = process_invoice(image_bytes)
                cleaned = result.replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned)
                validate_json(data)
                display_invoice(data)

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        st.session_state.processed_data = None

# Export Section
st.markdown("---")
with st.expander("📤 Export Options", expanded=True):
    if st.session_state.processed_data:
        col1, col2, col3 = st.columns(3)
        
        col1.download_button(
            "Download JSON",
            json.dumps(st.session_state.processed_data, indent=2),
            "invoice_data.json",
            "application/json"
        )
        
        csv = pd.DataFrame(st.session_state.processed_data['items']).to_csv(index=False)
        col2.download_button(
            "Download CSV",
            csv,
            "invoice_items.csv",
            "text/csv"
        )
        
        col3.download_button(
            "Download Preview",
            image_bytes if 'image_bytes' in locals() else b'',
            "document_preview.jpg",
            "image/jpeg"
        )

st.caption("ℹ️ Tip: Use the zoom slider for PDFs to improve preview clarity before extraction")
