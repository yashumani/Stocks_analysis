import fitz  # PyMuPDF
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Ensure the necessary NLTK data packages are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
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

def create_newsletter_image(text, sentiment_scores, output_path):
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
    
    # Save the image as a PNG file
    image.save(output_path)

def main(pdf_path, output_path):
    text = extract_text_from_pdf(pdf_path)
    sentiment_scores = perform_sentiment_analysis(text)
    
    create_newsletter_image(text, sentiment_scores, output_path)
    print(f"Newsletter image saved as {output_path}")

# Example usage
pdf_path = 'path/to/your/daily_or_quarterly_report.pdf'
output_path = 'path/to/output/newsletter.png'
main(pdf_path, output_path)
