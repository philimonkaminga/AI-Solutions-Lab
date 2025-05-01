# Invoice Extractor 🧾

An AI-powered document analysis system for extracting structured data from invoices (images & PDFs).
## Features ✨
- Process JPG/PNG images and scanned PDF invoices
- Extract key invoice data using GPT-4 Vision
- Validate extracted data structure
- Export results as JSON/CSV
- PDF first-page preview
- ZMW currency formatting

## Installation 💻

### Windows
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install poppler (PDF processing)
# Download from: https://github.com/oschwartz10612/poppler-windows/releases/
# Extract to: C:\Program Files (x86)\poppler-24.08.0
# Add to PATH: C:\Program Files (x86)\poppler-24.08.0\Library\bin
