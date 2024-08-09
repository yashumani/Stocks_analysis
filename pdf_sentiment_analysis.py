import streamlit as st
import camelot
import pandas as pd
from io import BytesIO

def extract_tables_from_pdf(pdf_file):
    # Read the PDF file as a byte stream
    pdf_data = BytesIO(pdf_file.read())
    
    # Extract tables from the PDF
    tables = camelot.read_pdf(pdf_data, pages='1-end', flavor='lattice')

    # Convert Camelot tables to pandas DataFrames
    return [table.df for table in tables]

def main():
    st.title("PDF Table Reader using Camelot")
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
