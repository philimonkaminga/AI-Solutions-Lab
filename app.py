import streamlit as st
import openai
import base64
import json
import pandas as pd
import pytesseract
import numpy as np
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import cv2

st.set_page_config(page_title="Invoice Expert Pro", page_icon="🧾", layout="centered")
st.title("🧾 Advanced Invoice Extractor")

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
    """Enhanced validation with arithmetic checks"""
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid invoice data format")
    
    # Field validation
    missing_keys = [key for key in REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Missing fields: {', '.join(missing_keys)}")
    
    # Items validation
    if not isinstance(data['items'], list) or len(data['items']) == 0:
        raise ValueError("Invalid items list")
    
    for item in data['items']:
        missing = [k for k in ITEM_KEYS if k not in item]
        if missing:
            raise ValueError(f"Item missing: {', '.join(missing)}")
        
        if not all(isinstance(item[k], (int, float)) for k in ['quantity', 'unit_price', 'total_price']):
            raise ValueError("Invalid numeric values in items")

    # Arithmetic validation
    calculated_total = data['subtotal'] + data['tax']
    if not abs(calculated_total - data['total']) < 0.01:
        raise ValueError(f"Total mismatch: {data['total']} vs calculated {calculated_total}")

def preprocess_image(image_bytes):
    """Advanced image preprocessing pipeline"""
    try:
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Sharpening
        img = img.filter(ImageFilter.SHARPEN)
        
        # Denoising
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Adaptive thresholding
        img = img.point(lambda x: 0 if x < 200 else 255, '1')
        
        # Save processed image
        processed_bytes = BytesIO()
        img.save(processed_bytes, format='JPEG', quality=100)
        return processed_bytes.getvalue()
        
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return image_bytes

def deskew_image(image):
    """Auto-deskew scanned documents using OpenCV"""
    try:
        image_np = np.array(image.convert('L'))
        thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_np, M, (w, h), 
                               flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(rotated)
    except:
        return image

def pdf_to_jpeg(pdf_bytes, zoom=3.0):
    """High-quality PDF conversion with deskewing"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        
        # Calculate DPI (300 DPI base)
        dpi = int(300 * zoom)
        mat = fitz.Matrix(dpi/72, dpi/72)
        
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Deskew image
        img = deskew_image(img)
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=100)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        st.error(f"PDF conversion error: {str(e)}")
        st.stop()

def process_invoice(image_bytes):
    """Hybrid OCR + Vision processing"""
    try:
        # Preprocess image
        processed_bytes = preprocess_image(image_bytes)
        img = Image.open(BytesIO(processed_bytes))
        
        # OCR Fallback
        ocr_text = pytesseract.image_to_string(img)
        
        base64_image = base64.b64encode(processed_bytes).decode('utf-8')
        
        messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": f"""Extract structured invoice data with:
                - invoice_number (string)
                - date (YYYY-MM-DD)
                - vendor (string)
                - items (array of {{description, quantity, unit_price, total_price}})
                - subtotal (number)
                - tax (number)
                - total (number)

                OCR Text (for cross-validation):
                {ocr_text[:2000]}

                Validation Rules:
                1. total MUST equal subtotal + tax
                2. item.total_price = quantity * unit_price
                3. Date format must be YYYY-MM-DD
                4. Numeric fields must be numbers

                Return ONLY valid JSON"""
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": messages}],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        raise

def display_invoice(data):
    """Interactive invoice display"""
    st.session_state.processed_data = data
    
    st.success("✅ Invoice data extracted successfully!")
    
    with st.expander("📄 Invoice Summary", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Invoice Number", data.get('invoice_number', 'N/A'))
        cols[1].metric("Date", data.get('date', 'N/A'))
        cols[2].metric("Vendor", data.get('vendor', 'N/A'))

    st.subheader("Itemized Products/Services")
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
        st.subheader("Financial Summary")
        cols = st.columns(3)
        cols[0].metric("Subtotal", f"ZMW{data['subtotal']:.2f}")
        cols[1].metric("Tax", f"ZMW{data['tax']:.2f}")
        cols[2].metric("Total", f"ZMW{data['total']:.2f}")

if uploaded_file and openai.api_key:
    try:
        image_bytes = None
        max_display_width = 1200
        
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, use_column_width=True)
            image_bytes = uploaded_file.getvalue()
            
        elif uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.getvalue()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                zoom_level = st.slider("PDF Zoom Level (1-10x)", 1.0, 10.0, 3.0, 0.1)
            with col2:
                if zoom_level > 6.0:
                    st.warning("High zoom may slow processing")
                
            image_bytes = pdf_to_jpeg(pdf_bytes, zoom=zoom_level)
            display_width = min(max_display_width, int(800 * zoom_level))
            st.image(image_bytes, width=display_width)

        if image_bytes and st.button("Analyze Invoice", type="primary"):
            with st.spinner("Performing deep analysis..."):
                result = process_invoice(image_bytes)
                cleaned = result.replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned)
                validate_json(data)
                display_invoice(data)

    except json.JSONDecodeError:
        st.error("Failed to parse API response")
        st.code(result if 'result' in locals() else "No response", language='json')
    except ValueError as ve:
        st.error(f"Validation error: {str(ve)}")
        st.code(cleaned if 'cleaned' in locals() else "No data", language='json')
    except Exception as e:
        st.error(f"Critical error: {str(e)}")

# Export Section
st.markdown("---")
with st.expander("📤 Export Data", expanded=True):
    if st.session_state.processed_data:
        cols = st.columns(3)
        
        cols[0].download_button(
            "Download JSON",
            json.dumps(st.session_state.processed_data, indent=2),
            "invoice_data.json",
            "application/json"
        )
        
        cols[1].download_button(
            "Download CSV",
            pd.DataFrame(st.session_state.processed_data['items']).to_csv(index=False),
            "invoice_items.csv",
            "text/csv"
        )
        
        cols[2].download_button(
            "Save Processed Image",
            image_bytes if 'image_bytes' in locals() else b'',
            "processed_document.jpg",
            "image/jpeg"
        )

st.caption("ℹ️ For best accuracy: Use 300+ DPI scans, clear layouts, and verify extracted totals")

# System requirements
with st.expander("⚙️ Configuration Guide"):
    st.markdown("""
    **System Requirements:**
    - Tesseract OCR installed
    - Poppler utilities (for PDF)
    - Minimum 4GB RAM
    
    **Installation:**
    ```bash
    # Linux
    sudo apt install tesseract-ocr poppler-utils
    
    # Mac
    brew install tesseract poppler
    
    # Windows: Download installers for Tesseract and Poppler
    ```
    """)

# import streamlit as st
# import openai
# import base64
# import json
# import pandas as pd
# from io import BytesIO
# import fitz  # PyMuPDF
# from PIL import Image

# st.set_page_config(page_title="Invoice Expert", page_icon="🧾", layout="centered")
# st.title("🧾 Professional Invoice Analyzer")

# # Initialize session state
# if 'processed_data' not in st.session_state:
#     st.session_state.processed_data = None

# # OpenAI API Key
# openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# # File uploader with supported formats
# uploaded_file = st.file_uploader("📤 Upload Invoice (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])

# # Invoice schema validation
# REQUIRED_KEYS = {
#     "invoice_number": "string",
#     "date": "string",
#     "vendor": "string",
#     "items": "array",
#     "subtotal": "number",
#     "tax": "number",
#     "total": "number"
# }

# ITEM_KEYS = ["description", "quantity", "unit_price", "total_price"]

# def validate_json(data):
#     """Validate invoice JSON structure"""
#     if not data or not isinstance(data, dict):
#         raise ValueError("Invalid invoice data format")
    
#     missing_keys = [key for key in REQUIRED_KEYS if key not in data]
#     if missing_keys:
#         raise ValueError(f"Missing fields: {', '.join(missing_keys)}")
    
#     if not isinstance(data['items'], list) or len(data['items']) == 0:
#         raise ValueError("Invalid items list")
    
#     for item in data['items']:
#         missing = [k for k in ITEM_KEYS if k not in item]
#         if missing:
#             raise ValueError(f"Item missing: {', '.join(missing)}")
        
#         if not all(isinstance(item[k], (int, float)) for k in ['quantity', 'unit_price', 'total_price']):
#             raise ValueError("Invalid numeric values in items")

# def process_invoice(image_bytes):
#     """Process invoice image using GPT-4 Vision"""
#     base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
#     messages = [{
#         "type": "image_url",
#         "image_url": {
#             "url": f"data:image/jpeg;base64,{base64_image}"
#         }
#     }]
    
#     messages.append({
#         "type": "text",
#         "text": """Extract structured invoice data as JSON with:
#         - invoice_number (string)
#         - date (string)
#         - vendor (string)
#         - items (array of {description, quantity, unit_price, total_price})
#         - subtotal (number)
#         - tax (number)
#         - total (number)
#         Return ONLY valid JSON without comments"""
#     })

#     response = openai.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[{"role": "user", "content": messages}],
#         max_tokens=1000
#     )
#     return response.choices[0].message.content.strip()

# def pdf_to_jpeg(pdf_bytes, zoom=2.0):
#     """Convert PDF to JPEG with adjustable zoom and memory management"""
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         page = doc.load_page(0)
        
#         # Show processing indicator for high zoom
#         if zoom > 5.0:
#             st.toast(f"Rendering at {zoom}x zoom...", icon="⏳")
            
#         mat = fitz.Matrix(zoom, zoom)
#         pix = page.get_pixmap(matrix=mat)
        
#         # Prevent excessive memory usage
#         if pix.width > 12000 or pix.height > 12000:
#             raise MemoryError("Image size exceeds maximum allowed dimensions (12,000px)")
            
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
#         # Optimize JPEG quality and compression
#         img_byte_arr = BytesIO()
#         img.save(img_byte_arr, format='JPEG', quality=90, optimize=True, progressive=True)
#         return img_byte_arr.getvalue()
        
#     except Exception as e:
#         st.error(f"PDF conversion error: {str(e)}")
#         st.stop()

# def display_invoice(data):
#     """Display extracted invoice data with enhanced layout"""
#     st.session_state.processed_data = data
    
#     st.success("✅ Invoice data extracted successfully!")
    
#     with st.container():
#         cols = st.columns(3)
#         cols[0].metric("Invoice Number", data.get('invoice_number', 'N/A'))
#         cols[1].metric("Date", data.get('date', 'N/A'))
#         cols[2].metric("Vendor", data.get('vendor', 'N/A'))

#     st.subheader("📦 Itemized List")
#     items_df = pd.DataFrame(data['items'])
#     items_df.index += 1
#     st.dataframe(
#         items_df.style.format({
#             'unit_price': 'ZMW{:.2f}',
#             'total_price': 'ZMW{:.2f}',
#             'quantity': '{:.0f}'
#         }),
#         use_container_width=True,
#         height=450
#     )

#     with st.container():
#         cols = st.columns(3)
#         cols[0].metric("Subtotal", f"ZMW{data['subtotal']:.2f}", 
#                       help="Total before taxes")
#         cols[1].metric("Tax", f"ZMW{data['tax']:.2f}", 
#                       help="Total tax amount")
#         cols[2].metric("Grand Total", f"ZMW{data['total']:.2f}", 
#                       help="Final payable amount")

# if uploaded_file and openai.api_key:
#     try:
#         image_bytes = None
#         max_display_width = 1200  # Maximum display width in pixels
        
#         if uploaded_file.type.startswith('image'):
#             st.image(uploaded_file, use_container_width=True)
#             image_bytes = uploaded_file.getvalue()
            
#         elif uploaded_file.type == "application/pdf":
#             pdf_bytes = uploaded_file.getvalue()
            
#             # Zoom controls with warnings
#             col1, col2 = st.columns([3, 1])
#             with col1:
#                 zoom_level = st.slider("PDF Zoom Level (1-10x)", 
#                                       1.0, 10.0, 2.0, 0.1,
#                                       help="Increase zoom for better text clarity")
#             with col2:
#                 if zoom_level > 5.0:
#                     st.warning("High zoom may slow processing")
                
#             image_bytes = pdf_to_jpeg(pdf_bytes, zoom=zoom_level)
            
#             # Calculate display width with upper limit
#             display_width = min(max_display_width, int(800 * zoom_level))
#             st.image(image_bytes, width=display_width)

#         if image_bytes and st.button("Extract Invoice Data", type="primary"):
#             with st.spinner("Analyzing document content..."):
#                 result = process_invoice(image_bytes)
#                 cleaned = result.replace('```json', '').replace('```', '').strip()
#                 data = json.loads(cleaned)
#                 validate_json(data)
#                 display_invoice(data)

#     except MemoryError as me:
#         st.error(f"Memory error: {str(me)}. Reduce zoom level and try again.")
#         st.session_state.processed_data = None
#     except Exception as e:
#         st.error(f"Processing error: {str(e)}")
#         st.session_state.processed_data = None

# # Export Section
# st.markdown("---")
# with st.expander("📁 Export Options", expanded=True):
#     if st.session_state.processed_data:
#         col1, col2, col3 = st.columns(3)
        
#         col1.download_button(
#             "Download JSON",
#             json.dumps(st.session_state.processed_data, indent=2),
#             "invoice_data.json",
#             "application/json",
#             help="Structured data in JSON format"
#         )
        
#         csv = pd.DataFrame(st.session_state.processed_data['items']).to_csv(index=False)
#         col2.download_button(
#             "Download CSV",
#             csv,
#             "invoice_items.csv",
#             "text/csv",
#             help="Item list in spreadsheet format"
#         )
        
#         col3.download_button(
#             "Save Preview",
#             image_bytes if 'image_bytes' in locals() else b'',
#             "document_preview.jpg",
#             "image/jpeg",
#             help="Download processed page image"
#         )

# st.caption("ℹ️ For best results: Use clear scans with visible numbers. Max file size: 20MB")

# # System requirements
# with st.expander("⚙️ System Recommendations"):
#     st.markdown("""
#     **Optimal performance tips:**
#     - Use zoom levels between 2x-5x for most documents
#     - Ensure documents have minimum 300 DPI resolution
#     - Process multi-page PDFs one page at a time
#     - Verify extracted amounts match document totals
#     """)
