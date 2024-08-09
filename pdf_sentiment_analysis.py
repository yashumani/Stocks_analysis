import fitz  # PyMuPDF
import streamlit as st

def extract_text_from_pdf(pdf_file):
    document = fitz.open("pdf", pdf_file.read())
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def main():
    st.title("Simple PDF Reader")
    st.write("Upload a PDF file to extract and display its text.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text", text, height=400)

if __name__ == '__main__':
    main()
