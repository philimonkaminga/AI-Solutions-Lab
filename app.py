import streamlit as st
import openai
from PIL import Image
import base64
import json

st.set_page_config(page_title="Simple Invoice OCR", page_icon="🧾", layout="centered")

st.title("🧾 Simple Invoice Extractor")

# OpenAI API Key
openai.api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an invoice image", type=["jpg", "jpeg", "png"])

# Handle image and GPT processing
if uploaded_file and openai.api_key:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice",  use_container_width=True)

    if st.button("Extract Invoice Info"):
        with st.spinner("Extrating Information..."):	
            try:
                # Convert image to base64
                image_bytes = uploaded_file.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{image_base64}"

                # Send request to GPT-4 turbo
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """
                                    Extract the following from this invoice image:
                                    - Invoice number
                                    - Date
                                    - Vendor
                                    - Items with description, quantity, unit price, and total price
                                    - Subtotal
                                    - Tax
                                    - Total

                                    Return result in JSON format only. No extra text or explanation.
                                    """ },
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=1000
                )

                result = response.choices[0].message.content.strip()
                st.markdown("### 📄 Extracted Data:")
                st.code(result, language="json")

                try:
                    st.json(json.loads(result))
                except:
                    st.warning("Could not parse the result as JSON.")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload an image and provide your OpenAI key.")

