import pandas as pd
from ftplib import FTP
from io import StringIO
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
from sklearn.model_selection import train_test_split
import numpy as np

# Ensure that the VADER lexicon is downloaded
nltk.download('vader_lexicon')

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

def fetch_yahoo_finance_headlines():
    headlines = []
    url = "https://finance.yahoo.com/rss/topstories"
    today_date = datetime.datetime.now().date()
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        for item in items:
            headline = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text
            # Correct ISO format for date parsing
            pub_date_parsed = datetime.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            description_html = item.description.text if item.description else headline
            description = BeautifulSoup(description_html, "html.parser").get_text()
            if headline and isinstance(headline, str) and pub_date_parsed.date() == today_date:
                headlines.append({"title": headline, "description": description, "url": link})
    except requests.RequestException as e:
        st.error(f"An error occurred while fetching Yahoo Finance news: {e}")

    return headlines

def fetch_google_news_headlines():
    headlines = []
    query = "NASDAQ"
    url = f"https://news.google.com/rss/search?q={query}"
    today_date = datetime.datetime.now().date()
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        for item in items:
            headline = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text
            pub_date_parsed = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            description_html = item.description.text if item.description else headline
            description = BeautifulSoup(description_html, "html.parser").get_text()
            if headline and isinstance(headline, str) and pub_date_parsed.date() == today_date:
                headlines.append({"title": headline, "description": description, "url": link})
    except requests.RequestException as e:
        st.error(f"An error occurred while fetching Google News: {e}")

    return headlines

def extract_ticker(headline, nasdaq_tickers):
    # Look for patterns like "(AAPL)" or standalone "AAPL"
    matches = re.findall(r'\b([A-Z]{1,5})\b', headline)
    for match in matches:
        if match in nasdaq_tickers:
            return match
    return None

def highlight_keywords(text, keywords):
    # Ensure text is a string
    if not text:
        return ""
    # Highlight keywords in the text using regex for whole words
    for keyword in keywords:
        text = re.sub(rf'\b{re.escape(keyword)}\b', f"**{keyword}**", text, flags=re.IGNORECASE)
    return text

def analyze_sentiment(text):
    # Perform sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def identify_tickers_and_companies(articles, nasdaq_tickers):
    # Identify NASDAQ tickers in articles
    identified_articles = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        ticker = extract_ticker(title, nasdaq_tickers)
        if ticker:
            identified_articles.append((article, [ticker]))

    return identified_articles

def train_predict_model(data):
    # Prepare data
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    # Feature selection
    features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']
    X = data[features]
    y = data['Close']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    next_5min_pred = model.predict(scaler.transform(X.iloc[-1].values.reshape(1, -1)))

    # Calculate error
    error = np.mean((predictions - y_test)**2)

    return predictions, y_test, error, next_5min_pred

def plot_stock_data(ticker, data):
    if data.empty:
        st.warning(f"No data available for {ticker} to display.")
        return

    # Ensure the index is a DatetimeIndex for mplfinance
    data.index = pd.to_datetime(data.index)
    
    # Add technical indicators
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    # Plotting
    apds = [mpf.make_addplot(data['EMA'], color='red', width=1.2),
            mpf.make_addplot(data['MACD'], color='green', width=1.2),
            mpf.make_addplot(data['Signal'], color='blue', width=1.2),
            mpf.make_addplot(data['VWAP'], color='yellow', width=1.2)]

    fig, axes = mpf.plot(data, type='candle', style='yahoo', volume=True,
                         title=f"{ticker} Stock Price", addplot=apds,
                         returnfig=True)

    # Enhance plot details
    axes[0].legend(['EMA', 'MACD', 'Signal Line', 'VWAP'])
    axes[0].set_ylabel('Price')
    axes[2].set_ylabel('Volume')
    axes[0].grid(True)
    fig.autofmt_xdate()  # Improve date readability

    st.pyplot(fig)


def display_articles_with_analysis(articles_with_tickers, period, interval):
    # Display all articles with sentiment analysis and identified tickers
    if not articles_with_tickers:
        st.warning("No relevant finance news articles found for today.")
        return

    sentiment_data = []

    for article, tickers in articles_with_tickers:
        title = article.get('title', '') or 'No title available'
        description = article.get('description', '') or 'No description available'
        sentiment = analyze_sentiment(description)  # Use description for sentiment analysis
        highlighted_title = highlight_keywords(title, tickers)
        highlighted_summary = highlight_keywords(description, tickers)
        
        # Add sentiment data for visualization
        for ticker in tickers:
            sentiment_data.append({
                'Ticker': ticker,
                'Positive': sentiment['pos'],
                'Negative': sentiment['neg'],
                'Neutral': sentiment['neu'],
                'Compound': sentiment['compound']
            })

        st.markdown(f"**Headline**: {highlighted_title}")
        st.markdown(f"**Summary**: {highlighted_summary}")  # Show description/summary
        st.markdown(f"**Tickers/Companies Identified**: **{', '.join(tickers)}**" if tickers else "No tickers identified.")
        st.markdown(f"**Link**: [Read more]({article['url']})")
        st.markdown(
            f"**Sentiment**: "
            f"Positive: **{sentiment['pos']:.2f}**, "
            f"Negative: **{sentiment['neg']:.2f}**, "
            f"Neutral: **{sentiment['neu']:.2f}**, "
            f"Compound: **{sentiment['compound']:.2f}**"
        )
        
        # Technical Analysis
        try:
            ticker_data = yf.download(tickers[0], period=period, interval=interval)  # Use selected period and interval
            plot_stock_data(tickers[0], ticker_data)
            predictions, y_test, error, next_5min_pred = train_predict_model(ticker_data)
            st.write(f"Prediction error: {error:.2f}")
            st.write(f"Next 5 min predicted price for {tickers[0]}: {next_5min_pred[0]:.2f}")
        except Exception as e:
            st.error(f"Failed to download data for {tickers[0]}: {e}")
        
        st.markdown("---")

def send_feedback_email(name, email, feedback):
    # Send feedback email using SMTP
    sender_email = "yashusharma800@gmail.com"  # Update this with your sender email
    sender_password = "sdoe qftq nrem uoql"  # Use the app password here
    receiver_email = "yashusharma800@gmail.com"

    subject = f"Feedback from {name}"
    body = f"Name: {name}\nEmail: {email}\n\nFeedback:\n{feedback}"

    # Create email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    # Send the email via SMTP server
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
    ("1hr","1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"),
    index=1  # Default to "5d"
)

interval = st.sidebar.selectbox(
    "Select Interval:",
    ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"),
    index=2  # Default to "5m"
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

# Fetch Latest Articles from Yahoo Finance and Google News
yahoo_articles = fetch_yahoo_finance_headlines()
google_articles = fetch_google_news_headlines()

# Combine and Filter Finance or Stock-Related News
all_articles = yahoo_articles + google_articles

# Step 3: Analyze News and Identify Tickers/Companies
articles_with_tickers = identify_tickers_and_companies(all_articles, nasdaq_tickers)

# Step 4: Display All Filtered Articles with Links and Analysis
display_articles_with_analysis(articles_with_tickers, period, interval)

# Feedback Form
st.sidebar.title("Feedback")
st.sidebar.markdown("We would love to hear your thoughts about the app!")

with st.sidebar.form("feedback_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    feedback = st.text_area("Your Feedback")
    submitted = st.form_submit_button("Submit")

    if submitted:
        send_feedback_email(name, email, feedback)