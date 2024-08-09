import os
import fitz  # PyMuPDF
from nltk.sentiment import SentimentIntensityAnalyzer
from PIL import Image, ImageDraw, ImageFont
import textwrap
import streamlit as st
import io

# Set the NLTK_DATA environment variable to use the local NLTK data directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path

def extract_text_from_pdf(pdf_file):
    document = fitz.open("pdf", pdf_file.read())
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def create_newsletter_image(text, sentiment_scores):
    width, height = 800, 1200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    margin = 40
    offset = 50
    for line in textwrap.wrap(text, width=70):
        draw.text((margin, offset), line, font=font, fill='black')
        offset += 15
    
    sentiment_text = f"Sentiment Analysis Scores: {sentiment_scores}"
    draw.text((margin, offset + 20), sentiment_text, font=font, fill='black')
    
    return image

def main():
    st.title("PDF Sentiment Analysis and Newsletter Creator")
    st.write("Upload a PDF file to extract text, perform sentiment analysis, and generate a newsletter image.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            text = extract_text_from_pdf(uploaded_file)
            sentiment_scores = perform_sentiment_analysis(text)
            image = create_newsletter_image(text, sentiment_scores)
            
            st.image(image, caption="Generated Newsletter", use_column_width=True)
            st.write("Sentiment Analysis Scores:", sentiment_scores)
            
            image_byte_arr = io.BytesIO()
            image.save(image_byte_arr, format='PNG')
            st.download_button(
                label="Download Newsletter Image",
                data=image_byte_arr.getvalue(),
                file_name="newsletter.png",
                mime="image/png"
            )

if __name__ == '__main__':
    main()
