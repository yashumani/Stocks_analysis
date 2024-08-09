import pandas as pd
import requests
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import concurrent.futures
import os
import logging
from functools import lru_cache
from transformers import pipeline
import plotly.express as px
from dotenv import load_dotenv
from ftplib import FTP
from io import StringIO
from dateutil import parser
# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure that the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Sentiment analysis using transformers
sentiment_pipeline = pipeline('sentiment-analysis')

@lru_cache(maxsize=32)
def get_nasdaq_tickers():
    # Connect to the NASDAQ FTP server and retrieve tickers
    ftp = FTP('ftp.nasdaqtrader.com')
    ftp.login()
    ftp.cwd('SymbolDirectory')
    data = []
    
    def handle_binary(data_bytes):
        data.append(data_bytes.decode('utf-8'))
    
    ftp.retrbinary('RETR nasdaqlisted.txt', handle_binary)
    ftp.quit()
    content = ''.join(data)
    file = StringIO(content)
    df = pd.read_csv(file, sep='|')
    return df[df['Test Issue'] == 'N']['Symbol'].astype(str).tolist()

def fetch_rss_feed(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return None

from dateutil import parser

def fetch_news_headlines():
    urls = [
        "https://finance.yahoo.com/rss/topstories",
        "https://news.google.com/rss/search?q=NASDAQ"
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_rss_feed, urls))
    
    headlines = []
    today_date = datetime.datetime.now().date()

    for content in results:
        if content:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")
            for item in items:
                headline = item.title.text
                link = item.link.text
                pub_date = item.pubDate.text
                try:
                    pub_date_parsed = parser.parse(pub_date)
                except ValueError as e:
                    st.warning(f"Error parsing date: {e}")
                    continue
                description_html = item.description.text if item.description else headline
                description = BeautifulSoup(description_html, "html.parser").get_text()
                if headline and isinstance(headline, str) and pub_date_parsed.date() == today_date:
                    headlines.append({"title": headline, "description": description, "url": link})

    return headlines

def extract_ticker(headline, nasdaq_tickers):
    matches = re.findall(r'\b([A-Z]{1,5})\b', headline)
    for match in matches:
        if match in nasdaq_tickers:
            return match
    return None

def highlight_keywords(text, keywords):
    if not text:
        return ""
    for keyword in keywords:
        text = re.sub(rf'\b{re.escape(keyword)}\b', f"**`{keyword}`**", text, flags=re.IGNORECASE)
    return text

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]

def identify_tickers_and_companies(articles, nasdaq_tickers):
    identified_articles = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        ticker = extract_ticker(title, nasdaq_tickers)
        if ticker:
            identified_articles.append((article, [ticker]))

    return identified_articles

def add_technical_indicators(data):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['MiddleBand'] = data['Close'].rolling(window=20).mean()
    data['UpperBand'] = data['MiddleBand'] + 1.96 * data['Close'].rolling(window=20).std()
    data['LowerBand'] = data['MiddleBand'] - 1.96 * data['Close'].rolling(window=20).std()

    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    return data

def train_predict_model(data):
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']
    X = data[features]
    y = data['Close']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    svr = SVR()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100], 'gamma': ['scale', 'auto']}

    grid_search = GridSearchCV(svr, parameters, cv=5)
    grid_search.fit(X_train, y_train)

    predictions = grid_search.predict(X_test)
    next_5min_pred = grid_search.predict(scaler.transform(X.iloc[-1].values.reshape(1, -1)))

    error = np.mean((predictions - y_test)**2)

    return predictions, y_test, error, next_5min_pred

def plot_interactive_stock_data(ticker, data):
    if data.empty:
        st.warning(f"No data available for {ticker} to display.")
        return

    data = add_technical_indicators(data)

    # Create a Plotly figure
    fig = px.line(data, x=data.index, y='Close', title=f"{ticker} Stock Prices Over Time")
    fig.add_scatter(x=data.index, y=data['EMA'], mode='lines', name='EMA')
    fig.add_scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD')
    fig.add_scatter(x=data.index, y=data['Signal'], mode='lines', name='Signal Line')
    fig.add_scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP')
    fig.add_scatter(x=data.index, y=data['UpperBand'], mode='lines', name='Upper Band', line=dict(dash='dash'))
    fig.add_scatter(x=data.index, y=data['MiddleBand'], mode='lines', name='Middle Band')
    fig.add_scatter(x=data.index, y=data['LowerBand'], mode='lines', name='Lower Band', line=dict(dash='dash'))

    # Use st.plotly_chart to display within Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
def display_articles_with_analysis(articles_with_tickers, period, interval):
    if not articles_with_tickers:
        st.warning("No relevant finance news articles found for today.")
        return

    sentiment_data = []

    for article, tickers in articles_with_tickers:
        title = article.get('title', '') or 'No title available'
        description = article.get('description', '') or 'No description available'
        sentiment = analyze_sentiment(description)
        highlighted_title = highlight_keywords(title, tickers)
        highlighted_summary = highlight_keywords(description, tickers)
        
        for ticker in tickers:
            sentiment_data.append({
                'Ticker': ticker,
                'Positive': sentiment['score'] if sentiment['label'] == 'POSITIVE' else 0,
                'Negative': sentiment['score'] if sentiment['label'] == 'NEGATIVE' else 0,
                'Neutral': 1 - sentiment['score'],
                'Compound': sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
            })

        st.markdown(f"**Headline**: {highlighted_title}")
        st.markdown(f"**Summary**: {highlighted_summary}")
        st.markdown(f"**Tickers/Companies Identified**: **`{', '.join(tickers)}`**" if tickers else "No tickers identified.")
        st.markdown(f"**Link**: [Read more]({article['url']})")
        st.markdown(
            f"**Sentiment**: "
            f"Positive: **`{sentiment['score']:.2f}`**" if sentiment['label'] == 'POSITIVE' else f"Negative: **`{sentiment['score']:.2f}`**"
        )
        
        try:
            ticker_data = yf.download(tickers[0], period=period, interval=interval)
            plot_interactive_stock_data(tickers[0], ticker_data)
            predictions, y_test, error, next_5min_pred = train_predict_model(ticker_data)
            st.write(f"Prediction error: {error:.2f}")
            st.write(f"Next 5 min predicted price for {tickers[0]}: {next_5min_pred[0]:.2f}")
        except Exception as e:
            st.error(f"Failed to download data for {tickers[0]}: {e}")
        
        st.markdown("---")

def send_feedback_email(name, email, feedback_type, feedback):
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    receiver_email = "yashusharma800@gmail.com"

    subject = f"Feedback from {name}"
    body = f"Name: {name}\nEmail: {email}\n\nFeedback Type: {feedback_type}\n\nFeedback:\n{feedback}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            st.success("Thank you for your feedback! Your response has been sent.")
    except Exception as e:
        st.error(f"Error sending feedback: {e}")

# Streamlit App
st.title("Today's US Finance News Analysis")

# Sidebar for date range and interval selection
st.sidebar.title("Chart Settings")
period = st.sidebar.selectbox(
    "Select Date Range:",
    ("1hr", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"),
    index=1
)

interval = st.sidebar.selectbox(
    "Select Interval:",
    ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"),
    index=2
)

# Sidebar Glossary
st.sidebar.title("Glossary & Instructions")
st.sidebar.markdown(
    """
    **How to Use the App:**
    1. **Fetch News**: The app automatically retrieves today's news articles related to finance, economy, stocks, markets, and business.
    2. **Analysis**: The app performs sentiment analysis on the fetched news articles and identifies relevant NASDAQ tickers.
    3. **Visualization**: View sentiment scores and highlighted information in the articles.

    **Reading the Sentiment Analysis:**
    - **Positive**: Indicates a positive sentiment score for the article summary.
    - **Negative**: Indicates a negative sentiment score for the article summary.
    - **Neutral**: Reflects a neutral sentiment score, showing balance.
    - **Compound**: This is a normalized score representing overall sentiment. Positive values suggest overall positive sentiment, while negative values suggest negative sentiment.

    **Highlighted Information:**
    - **Tickers/Keywords**: Keywords and tickers in headlines and summaries are highlighted to draw attention.

    **Feedback**: Use the feedback form to provide your thoughts about the app. Your feedback will be sent directly via email for further improvements.
    """
)

# Fetch NASDAQ tickers and company names automatically
nasdaq_tickers = get_nasdaq_tickers()

# Fetch Latest Articles from RSS Feeds
all_articles = fetch_news_headlines()

# Analyze News and Identify Tickers/Companies
articles_with_tickers = identify_tickers_and_companies(all_articles, nasdaq_tickers)

# Display All Filtered Articles with Links and Analysis
display_articles_with_analysis(articles_with_tickers, period, interval)

# Feedback Form
st.sidebar.title("Feedback")
st.sidebar.markdown("We would love to hear your thoughts about the app!")

with st.sidebar.form("feedback_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    feedback_type = st.selectbox("Feedback Type", ["General Feedback", "Bug Report", "Feature Request"])
    feedback = st.text_area("Your Feedback")
    submitted = st.form_submit_button("Submit")

    if submitted:
        send_feedback_email(name, email, feedback_type, feedback)
