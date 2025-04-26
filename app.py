import streamlit as st
import openai
from PIL import Image
import base64
import json
import pandas as pd
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from io import BytesIO

st.set_page_config(page_title="Document Analyzer", page_icon="📄", layout="centered")

st.title("📄 Document Analysis System")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# OpenAI API Key
openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# File uploader with expanded format support
uploaded_file = st.file_uploader("📤 Upload Document", type=["jpg", "jpeg", "png", "pdf"])

# Invoice schema validation with string type descriptors
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
    """Validate the structure and types of the JSON response"""
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid or empty data received")
    type_map = {
        "string": str,
        "number": (int, float),
        "array": list
    }
    
    # Check top-level keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required fields: {', '.join(missing_keys)}")
    
    # Check types
    for key, expected_type in REQUIRED_KEYS.items():
        actual_value = data.get(key)
        expected_py_type = type_map[expected_type]
        
        if not isinstance(actual_value, expected_py_type):
            raise ValueError(f"Field '{key}' should be {expected_type}, got {type(actual_value).__name__}")
    
    # Check items structure
    if not isinstance(data['items'], list):
        raise ValueError("Items should be a list")
    
    for i, item in enumerate(data['items']):
        missing_item_keys = [key for key in ITEM_KEYS if key not in item]
        if missing_item_keys:
            raise ValueError(f"Item {i+1} missing fields: {', '.join(missing_item_keys)}")
        
        # Check numeric fields
        for field in ['quantity', 'unit_price', 'total_price']:
            if not isinstance(item[field], (int, float)):
                raise ValueError(f"Item {i+1} {field} should be numeric")

def process_invoice(content, is_image=False):
    """Process invoice from image or text"""
    messages = []
    
    if is_image:
        image_url = f"data:image/jpeg;base64,{content}"
        messages.append({"type": "image_url", "image_url": {"url": image_url}})
    else:
        messages.append({"type": "text", "text": content})
    
    messages.append({"type": "text", "text": f"""
        Extract from this {'image' if is_image else 'document'} as JSON with:
        - invoice_number (string)
        - date (string)
        - vendor (string)
        - items (array of objects with description, quantity, unit_price, total_price)
        - subtotal (number)
        - tax (number)
        - total (number)
        Return ONLY the JSON object without any formatting or explanations.
    """})

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": messages}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def summarize_report(text):
    """Generate summary for text-based documents"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Summarize this document in 3-5 key points:\n{text[:3000]}"
        }]
    )
    return response.choices[0].message.content.strip()

def handle_pdf(uploaded_file):
    """Process PDF files"""
    try:
        # Try text extraction first
        pdf = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
        
        if len(text) > 100:  # Consider as text PDF
            return "report", text
        else:  # Process as image PDF
            images = convert_from_bytes(uploaded_file.getvalue())
            return "invoice", [image_to_base64(img) for img in images]
            
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return None, None

def image_to_base64(image):
    """Convert PIL image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if uploaded_file and openai.api_key:
    file_type = uploaded_file.type
    content = None
    doc_type = None

    if file_type in ["image/jpeg", "image/png"]:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            content = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            doc_type = "invoice"
        except Exception as e:
            st.error(f"❌ Image processing error: {str(e)}")
            st.stop()

    elif file_type == "application/pdf":
        doc_type, content = handle_pdf(uploaded_file)
        if not doc_type or not content:
            st.error("❌ Failed to process PDF file")
            st.stop()

    if st.button("Analyze Document"):
        with st.spinner("Processing..."):
            try:
                if doc_type == "invoice":
                    images = content if isinstance(content, list) else [content]
                    all_data = []
                    
                    for img in images:
                        result = process_invoice(img, is_image=True)
                        cleaned = result.replace('```json', '').replace('```', '').strip()
                        data = json.loads(cleaned)
                        validate_json(data)
                        all_data.append(data)
                    
                    # Store first invoice data in session state
                    st.session_state.processed_data = all_data[0]
                    
                    st.success("✅ Data extracted successfully!")
                    
                    # Header information
                    st.markdown("##### 📋 Invoice Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Invoice Number", st.session_state.processed_data.get("invoice_number", "N/A"))
                    col2.metric("Date", st.session_state.processed_data.get("date", "N/A"))
                    col3.metric("Vendor", st.session_state.processed_data.get("vendor", "N/A"))

                    # Items table
                    st.markdown("##### 🛒 Items")
                    items_df = pd.DataFrame(st.session_state.processed_data["items"])
                    items_df.index += 1
                    st.dataframe(
                        items_df.style.format({
                            'unit_price': 'ZMW{:.2f}',
                            'total_price': 'ZMW{:.2f}',
                            'quantity': '{:.0f}'
                        }),
                        use_container_width=True
                    )

                    # Totals
                    st.markdown("##### 💰 Totals")
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Subtotal", f"ZMW{st.session_state.processed_data['subtotal']:.2f}")
                    col5.metric("Tax", f"ZMW{st.session_state.processed_data['tax']:.2f}")
                    col6.metric("Total", f"ZMW{st.session_state.processed_data['total']:.2f}")

                    # Raw JSON expander
                    with st.expander("View Raw JSON"):
                        st.json(st.session_state.processed_data)

                elif doc_type == "report":
                    # Clear previous invoice data
                    st.session_state.processed_data = None
                    summary = summarize_report(content)
                    st.markdown("## 📝 Document Summary")
                    st.write(summary)
                    
                    with st.expander("View Full Text"):
                        st.text(content[:5000])

            except json.JSONDecodeError as e:
                st.error("❌ Failed to parse JSON response")
                st.code(result if 'result' in locals() else content[:500], language="text")
                st.stop()
            except ValueError as ve:
                st.error(f"❌ Validation error: {str(ve)}")
                st.code(cleaned if 'cleaned' in locals() else content[:500], language="json")
                st.stop()
            except Exception as e:
                st.session_state.processed_data = None
                st.error(f"❌ Processing error: {str(e)}")
                st.stop()

else:
    st.info("ℹ️ Please upload a document and provide your OpenAI key.")

# Footer with export options
st.markdown("---")
with st.expander("📤 Export Options"):
    if st.session_state.processed_data is not None:
        col7, col8, col9 = st.columns(3)
        
        # CSV Export
        csv = pd.DataFrame(st.session_state.processed_data['items']).to_csv(index=False).encode('utf-8')
        col7.download_button(
            label="Download Items as CSV",
            data=csv,
            file_name='invoice_items.csv',
            mime='text/csv',
        )
        
        # JSON Export
        json_data = json.dumps(st.session_state.processed_data, indent=2)
        col8.download_button(
            label="Download Full JSON",
            data=json_data,
            file_name='invoice_data.json',
            mime='application/json',
        )
        
        # HTML Summary
        html_report = f"""
        <html>
            <body>
                <h1>Invoice Report</h1>
                <p>Invoice Number: {st.session_state.processed_data['invoice_number']}</p>
                <p>Date: {st.session_state.processed_data['date']}</p>
                <p>Vendor: {st.session_state.processed_data['vendor']}</p>
                {pd.DataFrame(st.session_state.processed_data['items']).to_html()}
            </body>
        </html>
        """
        col9.download_button(
            label="Download HTML Report",
            data=html_report,
            file_name='invoice_report.html',
            mime='text/html',
        )
    else:
        st.warning("No data available for export")

st.markdown("---")
st.caption("ℹ️ Note: Accuracy depends on document quality and complexity. Always verify critical data.")

# Optional sample invoice
with st.expander("🖼️ Need a test invoice?"):
    st.markdown("Download a sample invoice image:")
    st.download_button(
        label="Download Sample Invoice",
        data=open("sample_invoice.jpg", "rb"),
        file_name="sample_invoice.jpg",
        mime="image/jpeg"
    )
