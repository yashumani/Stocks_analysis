import yfinance as yf
import pandas as pd
from ftplib import FTP
from io import StringIO
import datetime
import requests
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Ensure that the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize API key and request tracking
api_key = '15cca89c709e45ffabc5a04e7bba5841'  # Your NewsAPI key
initial_request_quota = 1000  # Example: set to the known daily limit for your plan

# Initialize session state to track requests
if 'request_count' not in st.session_state:
    st.session_state['request_count'] = 0

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

@st.cache_data(ttl=3600)
def fetch_latest_articles(max_articles=100):
    # Fetch the latest articles using pagination
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    all_articles = []
    page = 1
    page_size = 100  # Maximum articles per page allowed by NewsAPI for free tier

    while len(all_articles) < max_articles:
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': 'finance OR economy OR stock OR market OR business',
                'from': yesterday.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': page_size,
                'page': page,
                'apiKey': api_key
            }
        )

        data = response.json()
        
        # Update request count and estimate remaining requests
        st.session_state['request_count'] += 1
        remaining_requests = initial_request_quota - st.session_state['request_count']
        
        if data.get('status') == 'ok' and data.get('articles'):
            articles = data['articles']
            all_articles.extend(articles[:max_articles - len(all_articles)])
            
            if len(articles) < page_size:  # No more articles to fetch
                break

            page += 1
        else:
            st.error("Error fetching articles or no articles found.")
            break

    st.sidebar.info(f"Estimated API Requests Remaining Today: {remaining_requests}")
    st.write(f"Number of articles fetched: {len(all_articles)}")
    return all_articles

def highlight_keywords(text, keywords):
    # Ensure text is a string
    if not text:
        return ""
    # Highlight keywords in the text using regex for whole words
    for keyword in keywords:
        text = re.sub(rf'\b{re.escape(keyword)}\b', f"**`{keyword}`**", text, flags=re.IGNORECASE)
    return text

def filter_finance_news(articles):
    # Filter articles for finance-related keywords
    finance_keywords = ['finance', 'stock', 'market', 'NASDAQ', 'economy', 'investment', 'trading', 'Dow', 'S&P', 'Wall Street']
    filtered_articles = [
        article for article in articles 
        if any(
            keyword.lower() in (article.get('title', '') or '').lower() or 
            keyword.lower() in (article.get('description', '') or '').lower()
            for keyword in finance_keywords
        )
    ]
    st.write(f"Number of finance-related articles: {len(filtered_articles)}")
    return filtered_articles

def analyze_sentiment(text):
    # Perform sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def identify_tickers_and_companies(articles, nasdaq_tickers):
    # Identify NASDAQ tickers in articles
    identified_articles = []
    for article in articles:
        title = article.get('title', '') or ''
        content = title + ' ' + (article.get('description', '') or '')
        tickers = [ticker for ticker in nasdaq_tickers if re.search(rf'\b{re.escape(ticker)}\b', content)]
        if tickers:
            identified_articles.append((article, tickers))
    st.write(f"Identified tickers from articles: {[t for a, t in identified_articles]}")
    return identified_articles

@st.cache_data(ttl=3600)
def fetch_and_prepare_data(ticker, period='5d', interval='5m'):
    # Fetch high-frequency data for the selected period and interval
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        
        if stock_data.empty:
            st.warning(f"No data available for {ticker}.")
            return None

        # Create additional features
        stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
        stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
        stock_data['MACD'], stock_data['MACD_Signal'], _ = calculate_macd(stock_data['Close'])
        stock_data['VWAP'] = calculate_vwap(stock_data)

        stock_data.dropna(inplace=True)

        if stock_data.shape[0] < 1:
            st.warning(f"Insufficient data for {ticker} to perform analysis.")
            return None

        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_macd(close, short_window=12, long_window=26, signal_window=9):
    # Calculate MACD and MACD Signal
    exp1 = close.ewm(span=short_window, adjust=False).mean()
    exp2 = close.ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def calculate_vwap(data):
    # Calculate VWAP
    vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_rsi(data, window=14):
    # Calculate RSI
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_predict_model(stock_data):
    # Prepare features and target
    features = ['Close', 'EMA_20', 'EMA_50', 'MACD', 'VWAP']
    X = stock_data[features]
    y = stock_data['Close'].shift(-1).dropna()

    # Align features and target
    X = X.iloc[:-1]

    # Ensure there is data to fit the model
    if X.empty or y.empty:
        st.warning("Insufficient data to train the model.")
        return None, None, None, None

    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate error
    error = np.sqrt(mean_squared_error(y_test, predictions))

    # Transform the latest data point with the same features
    last_row = stock_data[features].iloc[-1:].values.reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    # Predict the next closing price
    next_5min_pred = model.predict(last_row_scaled)[0]

    return predictions, y_test, error, next_5min_pred

def detect_chart_patterns(stock_data):
    # Detect simple chart patterns like EMA crossover
    patterns = []
    if stock_data['EMA_20'].iloc[-1] > stock_data['EMA_50'].iloc[-1] and stock_data['EMA_20'].iloc[-2] <= stock_data['EMA_50'].iloc[-2]:
        patterns.append("Bullish Crossover (EMA 20 crosses above EMA 50)")
    if stock_data['EMA_20'].iloc[-1] < stock_data['EMA_50'].iloc[-1] and stock_data['EMA_20'].iloc[-2] >= stock_data['EMA_50'].iloc[-2]:
        patterns.append("Bearish Crossover (EMA 20 crosses below EMA 50)")
    return patterns

def display_articles_with_analysis(articles_with_tickers, period, interval):
    # Display all articles with sentiment analysis and identified tickers
    if not articles_with_tickers:
        st.warning("No relevant finance news articles with tickers found for today.")
        return

    sentiment_data = []

    for article, tickers in articles_with_tickers:
        title = article.get('title', '') or 'No title available'
        description = article.get('description', '') or 'No summary available'
        sentiment = analyze_sentiment(title)
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
        st.markdown(f"**Summary**: {highlighted_summary}")
        st.markdown(f"**Tickers/Companies Identified**: **`{', '.join(tickers)}`**")
        st.markdown(f"**Link**: [Read more]({article['url']})")
        st.markdown(
            f"**Sentiment**: "
            f"Positive: **`{sentiment['pos']:.2f}`**, "
            f"Negative: **`{sentiment['neg']:.2f}`**, "
            f"Neutral: **`{sentiment['neu']:.2f}`**, "
            f"Compound: **`{sentiment['compound']:.2f}`**"
        )
        
        # Perform technical analysis and prediction for each ticker
        for ticker in tickers:
            stock_data = fetch_and_prepare_data(ticker, period, interval)
            if stock_data is not None:
                result = train_predict_model(stock_data)
                if result is None:
                    continue

                predictions, y_test, error, next_5min_pred = result

                st.markdown(f"### Technical and Prediction Analysis for {ticker}")

                # Plotting with Matplotlib and mplfinance for price analysis
                fig, axlist = mpf.plot(
                    stock_data,
                    type='candle',
                    volume=False,  # Disable volume to separate analysis
                    title=f'Price Analysis for {ticker} ({period.capitalize()} Period)',
                    ylabel='Price',
                    style='yahoo',
                    mav=(20, 50),
                    addplot=[
                        mpf.make_addplot(stock_data['EMA_20'], color='green', label='EMA 20'),
                        mpf.make_addplot(stock_data['EMA_50'], color='blue', label='EMA 50'),
                        mpf.make_addplot(stock_data['VWAP'], color='purple', label='VWAP')
                    ],
                    returnfig=True
                )
                # Add legends manually
                axlist[0].legend(loc='upper left')
                st.pyplot(fig)

                # MACD Plot
                fig, ax = plt.subplots()
                ax.plot(stock_data.index, stock_data['MACD'], label='MACD', color='red')
                ax.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal Line', color='blue')
                ax.set_title(f'MACD Analysis for {ticker}')
                ax.set_xlabel('Date')
                ax.set_ylabel('MACD')
                ax.legend()
                st.pyplot(fig)

                # Detect chart patterns
                patterns = detect_chart_patterns(stock_data)
                if patterns:
                    st.markdown("**Detected Chart Patterns:**")
                    for pattern in patterns:
                        st.markdown(f"- {pattern}")

                st.markdown(f"Predicted Next Closing Price: **`{next_5min_pred:.2f}`**")
                st.markdown(f"Model RMSE: **`{error:.2f}`**")

                # Volume Analysis
                fig, ax = plt.subplots()
                ax.plot(stock_data.index, stock_data['Volume'], label='Volume', color='orange')
                ax.set_title(f'Volume Analysis for {ticker} ({period.capitalize()} Period)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Volume')
                ax.legend()
                st.pyplot(fig)

                st.markdown("---")

    # Plot sentiment comparison
    if sentiment_data:
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df = sentiment_df.groupby('Ticker').mean().reset_index()

        # Display all tickers analyzed
        st.write("Tickers Analyzed:", sentiment_df['Ticker'].tolist())

        # Matplotlib visualization for sentiment comparison
        fig, ax = plt.subplots()
        melted_df = sentiment_df.melt(id_vars='Ticker', var_name='Sentiment', value_name='Score')
        pivot_df = melted_df.pivot_table(index='Ticker', columns='Sentiment', values='Score', aggfunc='mean')
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title('Sentiment Analysis Comparison for Identified Tickers')
        ax.set_ylabel('Average Sentiment Score')
        st.pyplot(fig)

def send_feedback_email(name, email, feedback):
    # Send feedback email using SMTP
    sender_email = "your_email@example.com"  # Update this with your sender email
    sender_password = "your_app_password"  # Use the app password here
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
st.title("Today's US Finance News Analysis with Predictions")

# Sidebar: Date Range Selection
st.sidebar.title("Settings")
date_range = st.sidebar.selectbox(
    "Select Date Range for Analysis",
    ("1d", "5d", "1mo", "1y", "max"),
    index=1
)

# Sidebar Glossary
st.sidebar.title("Glossary & Instructions")
st.sidebar.markdown(
    """
    **How to Use the App:**
    1. **Select Date Range**: Choose the desired date range for the analysis from the options: Day, Week, Month, Year, and Max. The data granularity (interval) will automatically adjust based on your selection.
    
    2. **Fetch News**: The app automatically retrieves news articles related to finance, economy, stocks, markets, and business for the selected date range.

    3. **Analysis**: The app performs sentiment analysis on the fetched news articles and identifies relevant NASDAQ tickers.

    4. **Technical & Prediction Analysis**: For each identified ticker, the app conducts technical analysis, detects chart patterns, and predicts the next closing price.

    5. **Visualization**: View sentiment scores, technical indicators, and detected chart patterns in interactive charts.

    **Reading the Sentiment Analysis:**
    - **Positive**: Indicates a positive sentiment score for the article headline.
    - **Negative**: Indicates a negative sentiment score for the article headline.
    - **Neutral**: Reflects a neutral sentiment score, showing balance.
    - **Compound**: This is a normalized score representing overall sentiment. Positive values suggest overall positive sentiment, while negative values suggest negative sentiment.

    **Highlighted Information:**
    - **Tickers/Keywords**: Keywords and tickers in headlines and summaries are highlighted to draw attention.

    **Technical Indicators Explained:**
    - **EMA (Exponential Moving Average)**: Gives more weight to recent prices to identify trends.
    - **MACD (Moving Average Convergence Divergence)**: Indicates trend direction and strength, often used to signal buy/sell points.
    - **VWAP (Volume Weighted Average Price)**: Reflects the average price a stock has traded at throughout the day, based on volume and price.
    - **Chart Pattern Detection**: Identifies potential bullish or bearish crossovers using EMAs.

    **Feedback**: Use the feedback form to provide your thoughts about the app. Your feedback will be sent directly via email for further improvements.
    """
)

# Set interval based on date range
interval = "1d" if date_range == "max" else "5m" if date_range == "1d" else "15m" if date_range == "5d" else "1h"

# Fetch NASDAQ tickers automatically
nasdaq_tickers = get_nasdaq_tickers()

# Step 1: Fetch Latest Articles (up to 100, as per free API limit)
latest_articles = fetch_latest_articles(max_articles=100)

# Step 2: Filter Finance or Stock-Related News
finance_news = filter_finance_news(latest_articles)

# Step 3: Analyze News and Identify Tickers/Companies
articles_with_tickers = identify_tickers_and_companies(finance_news, nasdaq_tickers)

# Step 4: Display All Filtered Articles with Links and Analysis
display_articles_with_analysis(articles_with_tickers, period=date_range, interval=interval)

# Feedback Form
st.sidebar.title("Feedback")
st.sidebar.markdown("We would love to hear your thoughts about the app!!")

with st.sidebar.form("feedback_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    feedback = st.text_area("Your Feedback")
    submitted = st.form_submit_button("Submit")

    if submitted:
        send_feedback_email(name, email, feedback)
