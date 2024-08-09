import fitz  # PyMuPDF
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from PIL import Image, ImageDraw, ImageFont
import textwrap
import streamlit as st

# Ensure the necessary NLTK data packages are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def extract_text_from_pdf(pdf_file):
    # Open the PDF file from the uploaded bytes
    document = fitz.open("pdf", pdf_file.read())
    text = ""

    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text

def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def create_newsletter_image(text, sentiment_scores):
    # Create an image with white background
    width, height = 800, 1200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Load a font
    font = ImageFont.load_default()
    
    # Define text wrapping
    margin = 40
    offset = 50
    for line in textwrap.wrap(text, width=70):
        draw.text((margin, offset), line, font=font, fill='black')
        offset += 15
    
    # Add sentiment analysis scores at the bottom
    sentiment_text = f"Sentiment Analysis Scores: {sentiment_scores}"
    draw.text((margin, offset + 20), sentiment_text, font=font, fill='black')
    
    return image

def main():
    st.title("PDF Sentiment Analysis and Newsletter Creator")
    st.write("Upload a PDF file to extract text, perform sentiment analysis, and generate a newsletter image.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Extract text from the uploaded PDF
            text = extract_text_from_pdf(uploaded_file)
            
            # Perform sentiment analysis
            sentiment_scores = perform_sentiment_analysis(text)
            
            # Create a newsletter image
            image = create_newsletter_image(text, sentiment_scores)
            
            # Display results
            st.image(image, caption="Generated Newsletter", use_column_width=True)
            st.write("Sentiment Analysis Scores:", sentiment_scores)
            
            # Download the image
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
