import streamlit as st
import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_file):
    # Open the PDF file using pdfplumber
    with pdfplumber.open(pdf_file) as pdf:
        tables = []
        # Iterate through each page in the PDF
        for page in pdf.pages:
            # Extract tables from each page
            page_tables = page.extract_tables()
            for table in page_tables:
                # Convert each table to a pandas DataFrame
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

def main():
    st.title("PDF Table Reader using pdfplumber")
    st.write("Upload a PDF file to extract and display its tables.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting tables..."):
            tables = extract_tables_from_pdf(uploaded_file)
            if tables:
                st.success("Tables extracted successfully!")
                for i, table in enumerate(tables):
                    st.write(f"Table {i+1}")
                    st.dataframe(table)
            else:
                st.warning("No tables found in the PDF.")

if __name__ == '__main__':
    main()
