import streamlit as st
import tabula
import pandas as pd
from io import BytesIO

def extract_tables_from_pdf(pdf_file):
    # Read the PDF file as a byte stream
    pdf_data = BytesIO(pdf_file.read())
    
    # Extract tables from the PDF
    tables = tabula.read_pdf(pdf_data, pages='all', multiple_tables=True, lattice=True)

    return tables

def main():
    st.title("PDF Table Reader")
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
