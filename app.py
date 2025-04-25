import streamlit as st
import openai
from PIL import Image
import base64
import json
import pandas as pd

st.set_page_config(page_title="Simple Invoice OCR", page_icon="🧾", layout="centered")

st.title("🧾 Simple Invoice Extractor")

# OpenAI API Key
openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an invoice image", type=["jpg", "jpeg", "png"])

# JSON schema for validation
REQUIRED_KEYS = {
    "invoice_number": str,
    "date": str,
    "vendor": str,
    "items": list,
    "subtotal": float,
    "tax": float,
    "total": float
}

ITEM_KEYS = ["description", "quantity", "unit_price", "total_price"]

def validate_json(data):
    """Validate the structure and types of the JSON response"""
    # Check top-level keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required fields: {', '.join(missing_keys)}")
    
    # Check item structure
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

if uploaded_file and openai.api_key:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice", use_container_width=True)

    if st.button("Extract Invoice Info"):
        with st.spinner("Extracting Information..."):
            try:
                # Convert image to base64
                image_bytes = uploaded_file.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{image_base64}"

                # Structured prompt with JSON example
                prompt = f"""Analyze this invoice image and extract structured data. 
                Return ONLY a JSON object with EXACTLY these fields:
                {json.dumps({k: "<type>" if isinstance(v, type) else v for k, v in REQUIRED_KEYS.items()}, indent=2)}
                Items should be a list of objects with: {', '.join(ITEM_KEYS)}
                Ensure all numbers are numeric types, not strings.
                NO additional text or formatting - ONLY the JSON object."""

                # API call
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=1000
                )

                raw_response = response.choices[0].message.content.strip()
                
                # Clean and parse response
                cleaned_response = raw_response.replace('```json', '').replace('```', '').strip()
                
                try:
                    data = json.loads(cleaned_response)
                    validate_json(data)
                    
                    # Display results
                    st.success("✅ Data extracted successfully!")
                    
                    # Header information
                    st.markdown("##### 📋 Invoice Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Invoice Number", data.get("invoice_number", "N/A"))
                    col2.metric("Date", data.get("date", "N/A"))
                    col3.metric("Vendor", data.get("vendor", "N/A"))

                    # Items table
                    st.markdown("##### 🛒 Items")
                    items_df = pd.DataFrame(data["items"])
                    items_df.index += 1  # Start index at 1
                    st.dataframe(
                        items_df.style.format({
                            'unit_price': 'ZMW{:.2f}',
                            'total_price': 'ZMW{:.2f}',
                            'quantity': '{:.0f}'
                        }),
                        use_container_width=True
                    )

                    # Totals
                    st.markdown("##### Totals")
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Subtotal", f"ZMW{data['subtotal']:.2f}")
                    col5.metric("Tax", f"ZMW{data['tax']:.2f}")
                    col6.metric("Total", f"ZMW{data['total']:.2f}")

                    # Raw JSON expander
                    with st.expander("View Raw JSON"):
                        st.json(data)

                except json.JSONDecodeError:
                    st.error("❌ Failed to parse JSON response. Raw GPT output:")
                    st.code(raw_response, language="text")
                    st.stop()
                except ValueError as ve:
                    st.error(f"❌ Validation error: {str(ve)}")
                    st.code(cleaned_response, language="json")
                    st.stop()

            except openai.error.OpenAIError as e:
                st.error(f"❌ OpenAI API Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

else:
    st.info("ℹ️ Please upload an image and provide your OpenAI key.")

# Footer with export options and disclaimer
st.markdown("---")
with st.expander("📤 Export Options"):
    if 'data' in locals():
        col7, col8, col9 = st.columns(3)
        # CSV Export
        csv = pd.DataFrame(data['items']).to_csv(index=False).encode('utf-8')
        col7.download_button(
            label="Download Items as CSV",
            data=csv,
            file_name='invoice_items.csv',
            mime='text/csv',
        )
        
        # JSON Export
        json_data = json.dumps(data, indent=2)
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
                <p>Invoice Number: {data['invoice_number']}</p>
                <p>Date: {data['date']}</p>
                <p>Vendor: {data['vendor']}</p>
                {pd.DataFrame(data['items']).to_html()}
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
st.caption("ℹ️ Note: Accuracy depends on image quality and invoice complexity. Always verify critical data.")

# Optional: Add a sample image for testing
with st.expander("🖼️ Need a test invoice?"):
    st.markdown("Download a sample invoice image:")
    st.download_button(
        label="Download Sample Invoice",
        data=open("sample_invoice.jpg", "rb"),
        file_name="sample_invoice.jpg",
        mime="image/jpeg"
    )


# import streamlit as st
# import openai
# from PIL import Image
# import base64
# import json

# st.set_page_config(page_title="Simple Invoice OCR", page_icon="🧾", layout="centered")

# st.title("🧾 Simple Invoice Extractor")

# # OpenAI API Key
# openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# # Upload Image
# uploaded_file = st.file_uploader("📤 Upload an invoice image", type=["jpg", "jpeg", "png"])

# # Handle image and GPT processing
# if uploaded_file and openai.api_key:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Invoice",  use_container_width=True)

#     if st.button("Extract Invoice Info"):
#         with st.spinner("Extrating Information..."):	
#             try:
#                 # Convert image to base64
#                 image_bytes = uploaded_file.getvalue()
#                 image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#                 image_url = f"data:image/jpeg;base64,{image_base64}"

#                 # Send request to GPT-4 turbo
#                 response = openai.chat.completions.create(
#                     model="gpt-4-turbo",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": """
#                                     Extract the following from this invoice image:
#                                     - Invoice number
#                                     - Date
#                                     - Vendor
#                                     - Items with description, quantity, unit price, and total price
#                                     - Subtotal
#                                     - Tax
#                                     - Total

#                                     Return result in JSON format only. No extra text or explanation.
#                                     """ },
#                                 {"type": "image_url", "image_url": {"url": image_url}}
#                             ]
#                         }
#                     ],
#                     max_tokens=1000
#                 )

#                 result = response.choices[0].message.content.strip()
#                 st.markdown("### 📄 Extracted Data:")
#                 st.code(result, language="json")

#                 try:
#                     st.json(json.loads(result))
#                 except:
#                     st.warning("Could not parse the result as JSON.")

#             except Exception as e:
#                 st.error(f"Error: {e}")
# else:
#     st.info("Please upload an image and provide your OpenAI key.")

