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
import pandas_ta as ta  # Replaced `ta` with `pandas-ta`

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
        ftp = FTP('ftp.nasdaqtrader.com')
        ftp.login()
        ftp.cwd('SymbolDirectory')

        def fetch_tickers(filename):
            data = []
            def handle_binary(data_bytes):
                data.append(data_bytes.decode('utf-8'))
            ftp.retrbinary(f'RETR {filename}', handle_binary)
            content = ''.join(data)
            file = StringIO(content)
            return pd.read_csv(file, sep='|')

        nasdaq_df = fetch_tickers('nasdaqlisted.txt')
        nasdaq_df.columns = nasdaq_df.columns.str.strip()
        nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']
        nasdaq_df['Exchange'] = 'NASDAQ'

        other_df = fetch_tickers('otherlisted.txt')
        other_df.columns = other_df.columns.str.strip()
        other_df = other_df[other_df['Test Issue'] == 'N']
        other_df['Exchange'] = other_df['Exchange'].map({'A': 'AMEX', 'N': 'NYSE'})

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
    
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['50_SMA'] = data['Close'].rolling(window=50).mean()
    
    # Calculating RSI using pandas-ta
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
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

# Function to apply risk management
def apply_risk_management(stock, buy_signals, sell_signals, data):
    positions = []
    stop_loss = 0.95  # Example: 5% below buy price
    take_profit = 1.10  # Example: 10% above buy price

    for i in range(len(data)):
        if buy_signals[i]:
            buy_price = data['Close'].iloc[i]
            positions.append(('Buy', buy_price, data.index[i]))
        elif sell_signals[i]:
            if len(positions) > 0 and positions[-1][0] == 'Buy':
                sell_price = data['Close'].iloc[i]
                positions.append(('Sell', sell_price, data.index[i]))
        elif len(positions) > 0 and positions[-1][0] == 'Buy':
            current_price = data['Close'].iloc[i]
            if current_price <= positions[-1][1] * stop_loss:
                positions.append(('Sell (Stop Loss)', current_price, data.index[i]))
            elif current_price >= positions[-1][1] * take_profit:
                positions.append(('Sell (Take Profit)', current_price, data.index[i]))

    return positions

# Function to plot data
def plot_data(data, stock, buy_signals, sell_signals):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['20_SMA'], label='20 Day SMA')
    plt.plot(data['50_SMA'], label='50 Day SMA')

    plt.scatter(data.index[buy_signals], data['Close'][buy_signals], label='Buy Signal', marker='^', color='green')
    plt.scatter(data.index[sell_signals], data['Close'][sell_signals], label='Sell Signal', marker='v', color='red')

    plt.title(f'{stock} Stock Price and Signals')
    plt.legend()
    plt.show()

# Main function for the Streamlit app
def main():
    st.title("Stock Analysis Dashboard")

    all_tickers_df = get_us_stock_tickers()
    if all_tickers_df.empty:
        st.error("Failed to load US stock ticker data.")
        return

    ticker_set = set(all_tickers_df['Symbol'])

    news_articles = [
        "https://finance.yahoo.com/quote/AAPL/news/",
        "https://finance.yahoo.com/quote/MSFT/news/",
    ]

    extracted_tickers = []
    for url in news_articles:
        content = fetch_article_content(url)
        if content:
            extracted_tickers += extract_tickers_from_text(content, ticker_set)
    
    extracted_tickers = list(set(extracted_tickers))  # Remove duplicates

    if not extracted_tickers:
        st.error("No tickers found in the news articles.")
        return
    
    st.write("Extracted Tickers from News Articles:", extracted_tickers)

    selected_ticker = st.selectbox("Select Ticker for Analysis", extracted_tickers)

    if selected_ticker is None or selected_ticker == "":
        st.error("No ticker selected.")
        return

    st.write(f"You selected: {selected_ticker}")

    company_profile = fetch_company_info(selected_ticker)
    if company_profile:
        st.header(f"Company Profile for {company_profile['Name']}")
        st.write(f"**Sector:** {company_profile['Sector']}")
        st.write(f"**Industry:** {company_profile['Industry']}")
        st.write(f"**Market Cap:** {company_profile['Market Cap']}")
        st.write(f"**Full Time Employees:** {company_profile['Full Time Employees']}")
        st.write(f"**Description:** {company_profile['Description']}")

    technical_data = perform_technical_analysis(selected_ticker)
    st.header("Technical Analysis")
    st.line_chart(technical_data[['Close', '20_SMA', '50_SMA']])
    st.line_chart(technical_data[['RSI']])

    st.subheader("Stock Predictions")
    buy_signals = technical_data['RSI'] < 30
    sell_signals = technical_data['RSI'] > 70

    positions = apply_risk_management(selected_ticker, buy_signals, sell_signals, technical_data)
    for position in positions:
        st.write(f"Action: {position[0]}, Price: {position[1]}, Date: {position[2]}")

    plot_data(technical_data, selected_ticker, buy_signals, sell_signals)

if __name__ == "__main__":
    main()
