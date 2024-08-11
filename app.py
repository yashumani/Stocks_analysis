# Streamlit App for Stock Analysis Dashboard

import streamlit as st
import pandas as pd
import yfinance as yf
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ftplib import FTP
from io import StringIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import spacy
import numpy as np

# Initialize NLP models
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Configure the Streamlit app
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Function to fetch tickers from US stock exchanges
def get_us_stock_tickers():
    try:
        # Connect to the NASDAQ FTP server
        ftp = FTP('ftp.nasdaqtrader.com')
        ftp.login()
        ftp.cwd('SymbolDirectory')

        # Function to download and read the data
        def fetch_tickers(filename):
            data = []
            def handle_binary(data_bytes):
                data.append(data_bytes.decode('utf-8'))
            ftp.retrbinary(f'RETR {filename}', handle_binary)
            content = ''.join(data)
            file = StringIO(content)
            return pd.read_csv(file, sep='|')

        # Fetch tickers from NASDAQ
        nasdaq_df = fetch_tickers('nasdaqlisted.txt')
        nasdaq_df.columns = nasdaq_df.columns.str.strip()
        nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']
        nasdaq_df['Exchange'] = 'NASDAQ'

        # Fetch tickers from NYSE and AMEX
        other_df = fetch_tickers('otherlisted.txt')
        other_df.columns = other_df.columns.str.strip()
        other_df = other_df[other_df['Test Issue'] == 'N']
        other_df['Exchange'] = other_df['Exchange'].map({'A': 'AMEX', 'N': 'NYSE'})

        # Combine all tickers into a single DataFrame
        nasdaq_df = nasdaq_df[['Symbol', 'Security Name', 'Exchange']]
        other_df = other_df[['ACT Symbol', 'Security Name', 'Exchange']]
        nasdaq_df.columns = ['Symbol', 'Security Name', 'Exchange']
        other_df.columns = ['Symbol', 'Security Name', 'Exchange']

        all_tickers = pd.concat([nasdaq_df, other_df])
        ftp.quit()
        return all_tickers

    except Exception as e:
        st.error(f"An error occurred while fetching tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Security Name', 'Exchange'])

# Function to extract potential stock tickers from news articles
def extract_tickers_from_text(text, known_tickers):
    doc = nlp(text)
    extracted_tickers = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            candidate = ent.text.upper()
            if candidate in known_tickers:
                extracted_tickers.add(candidate)
    return list(extracted_tickers)

# Function to fetch detailed company information using yfinance
def fetch_company_info(ticker):
    company_info = yf.Ticker(ticker)
    try:
        info = company_info.info
        profile = {
            "Name": info.get('longName', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A'),
            "Full Time Employees": info.get('fullTimeEmployees', 'N/A'),
            "Description": info.get('longBusinessSummary', 'N/A')
        }
        return profile
    except Exception as e:
        st.error(f"An error occurred while fetching company info: {e}")
        return None

# Function to perform technical analysis
def perform_technical_analysis(ticker):
    data = yf.download(ticker, period="1y")
    
    # Calculate technical indicators like SMA, RSI, etc.
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Function to fetch and parse article content using newspaper3k
def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"An error occurred while fetching article content: {e}"

# Function to perform sentiment analysis on text
def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound'], 'Positive' if score['compound'] > 0.05 else 'Negative' if score['compound'] < -0.05 else 'Neutral'

# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Placeholder for predictions (can be extended with ML models)
def generate_predictions(technical_data):
    # Use technical indicators and historical data for basic prediction logic
    recent_rsi = technical_data['RSI'].iloc[-1]
    prediction = "Hold"
    if recent_rsi < 30:
        prediction = "Buy"
    elif recent_rsi > 70:
        prediction = "Sell"
    return prediction

# Main function for the Streamlit app
def main():
    st.title("Stock Analysis Dashboard")

    # Fetch US stock tickers
    all_tickers_df = get_us_stock_tickers()
    if all_tickers_df.empty:
        st.error("Failed to load US stock ticker data.")
        return

    ticker_set = set(all_tickers_df['Symbol'])

    # Fetch and display news articles
    st.subheader("News Articles")
    try:
        # Use yfinance to get news articles for a specific ticker
        ticker_symbol = "AAPL"
        company_info = yf.Ticker(ticker_symbol)
        news = company_info.news
        extracted_tickers = []

        if news:
            for article in news[:5]:  # Display the top 5 news articles
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Published:** {article['providerPublishTime']}")
                st.write(f"[Read more]({article['link']})")

                # Fetch article content
                content = fetch_article_content(article['link'])
                if content.strip():  # Check if content is not empty
                    st.write(content[:500] + "...")  # Display the first 500 characters

                    # Sentiment analysis
                    score, sentiment = analyze_sentiment(content)
                    st.write(f"**Sentiment Score:** {score:.2f} ({sentiment})")

                    # Extract tickers from content
                    tickers_in_article = extract_tickers_from_text(content, ticker_set)
                    extracted_tickers.extend(tickers_in_article)
                else:
                    st.write("No content found.")
                st.write("---")

            # Generate word cloud for the headlines
            headlines = ' '.join([article['title'] for article in news[:5]])
            st.write("### Word Cloud of News Headlines")
            generate_wordcloud(headlines)

            # Show extracted tickers
            extracted_tickers = list(set(extracted_tickers))  # Remove duplicates
            st.write("Extracted Tickers from News Articles:", extracted_tickers)

        else:
            st.write("No recent news found.")
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")

    # Allow users to select tickers for analysis
    selected_ticker = st.selectbox("Select Ticker for Analysis", extracted_tickers)
    st.write(f"You selected: {selected_ticker}")

    # Fetch detailed company information
    company_profile = fetch_company_info(selected_ticker)
    if company_profile:
        st.header(f"Company Profile for {company_profile['Name']}")
        st.write(f"**Sector:** {company_profile['Sector']}")
        st.write(f"**Industry:** {company_profile['Industry']}")
        st.write(f"**Market Cap:** {company_profile['Market Cap']}")
        st.write(f"**Full Time Employees:** {company_profile['Full Time Employees']}")
        st.write(f"**Description:** {company_profile['Description']}")

    # Perform technical analysis
    technical_data = perform_technical_analysis(selected_ticker)
    st.header("Technical Analysis")
    st.line_chart(technical_data[['Close', 'SMA_20', 'SMA_50']])
    st.line_chart(technical_data[['RSI']])

    # Generate predictions
    st.subheader("Stock Predictions")
    prediction = generate_predictions(technical_data)
    st.write(f"Based on the RSI value, the suggested action is: **{prediction}**")

if __name__ == "__main__":
    main()
