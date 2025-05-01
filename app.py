

import streamlit as st
import openai
import base64
import json
import pandas as pd
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Invoice Expert", page_icon="🧾", layout="centered")
st.title("🧾 Professional Invoice Analyzer")

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
    """Convert PDF to JPEG with adjustable zoom and memory management"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        
        # Show processing indicator for high zoom
        if zoom > 5.0:
            st.toast(f"Rendering at {zoom}x zoom...", icon="⏳")
            
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Prevent excessive memory usage
        if pix.width > 12000 or pix.height > 12000:
            raise MemoryError("Image size exceeds maximum allowed dimensions (12,000px)")
            
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Optimize JPEG quality and compression
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90, optimize=True, progressive=True)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        st.error(f"PDF conversion error: {str(e)}")
        st.stop()

def display_invoice(data):
    """Display extracted invoice data with enhanced layout"""
    st.session_state.processed_data = data
    
    st.success("✅ Invoice data extracted successfully!")
    
    with st.container():
        cols = st.columns(3)
        cols[0].metric("Invoice Number", data.get('invoice_number', 'N/A'))
        cols[1].metric("Date", data.get('date', 'N/A'))
        cols[2].metric("Vendor", data.get('vendor', 'N/A'))

    st.subheader("📦 Itemized List")
    items_df = pd.DataFrame(data['items'])
    items_df.index += 1
    st.dataframe(
        items_df.style.format({
            'unit_price': 'ZMW{:.2f}',
            'total_price': 'ZMW{:.2f}',
            'quantity': '{:.0f}'
        }),
        use_container_width=True,
        height=450
    )

    with st.container():
        cols = st.columns(3)
        cols[0].metric("Subtotal", f"ZMW{data['subtotal']:.2f}", 
                      help="Total before taxes")
        cols[1].metric("Tax", f"ZMW{data['tax']:.2f}", 
                      help="Total tax amount")
        cols[2].metric("Grand Total", f"ZMW{data['total']:.2f}", 
                      help="Final payable amount")

if uploaded_file and openai.api_key:
    try:
        image_bytes = None
        max_display_width = 1200  # Maximum display width in pixels
        
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, use_container_width=True)
            image_bytes = uploaded_file.getvalue()
            
        elif uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            
            # Zoom controls with warnings
            col1, col2 = st.columns([3, 1])
            with col1:
                zoom_level = st.slider("PDF Zoom Level (1-10x)", 
                                      1.0, 10.0, 2.0, 0.1,
                                      help="Increase zoom for better text clarity")
            with col2:
                if zoom_level > 5.0:
                    st.warning("High zoom may slow processing")
                
            image_bytes = pdf_to_jpeg(pdf_bytes, zoom=zoom_level)
            
            # Calculate display width with upper limit
            display_width = min(max_display_width, int(800 * zoom_level))
            st.image(image_bytes, width=display_width)

        if image_bytes and st.button("Extract Invoice Data", type="primary"):
            with st.spinner("Analyzing document content..."):
                result = process_invoice(image_bytes)
                cleaned = result.replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned)
                validate_json(data)
                display_invoice(data)

    except MemoryError as me:
        st.error(f"Memory error: {str(me)}. Reduce zoom level and try again.")
        st.session_state.processed_data = None
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.session_state.processed_data = None

# Export Section
st.markdown("---")
with st.expander("📁 Export Options", expanded=True):
    if st.session_state.processed_data:
        col1, col2, col3 = st.columns(3)
        
        col1.download_button(
            "Download JSON",
            json.dumps(st.session_state.processed_data, indent=2),
            "invoice_data.json",
            "application/json",
            help="Structured data in JSON format"
        )
        
        csv = pd.DataFrame(st.session_state.processed_data['items']).to_csv(index=False)
        col2.download_button(
            "Download CSV",
            csv,
            "invoice_items.csv",
            "text/csv",
            help="Item list in spreadsheet format"
        )
        
        col3.download_button(
            "Save Preview",
            image_bytes if 'image_bytes' in locals() else b'',
            "document_preview.jpg",
            "image/jpeg",
            help="Download processed page image"
        )

st.caption("ℹ️ For best results: Use clear scans with visible numbers. Max file size: 20MB")

# System requirements
with st.expander("⚙️ System Recommendations"):
    st.markdown("""
    **Optimal performance tips:**
    - Use zoom levels between 2x-5x for most documents
    - Ensure documents have minimum 300 DPI resolution
    - Process multi-page PDFs one page at a time
    - Verify extracted amounts match document totals
    """)
