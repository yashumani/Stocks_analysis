"""
Author: Yashu
Date: August 1, 2024

Summary:
This Streamlit app fetches the latest financial news headlines relevant to NASDAQ stocks,
performs sentiment analysis on the headlines and article content, identifies potential stock gainers based on positive sentiment scores,
and allows users to perform technical analysis on the selected stocks.
"""

import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import nltk
from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import mplfinance as mpf
from ftplib import FTP
from io import StringIO

# Ensure the NLTK VADER lexicon is downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch the list of NASDAQ tickers
def get_nasdaq_tickers():
    # Connect to the NASDAQ FTP server
    ftp = FTP('ftp.nasdaqtrader.com')
    ftp.login()

    # Navigate to the directory
    ftp.cwd('SymbolDirectory')

    # Define a function to handle the downloaded data
    data = []

    def handle_binary(data_bytes):
        data.append(data_bytes.decode('utf-8'))

    # Retrieve the file
    ftp.retrbinary('RETR nasdaqlisted.txt', handle_binary)
    ftp.quit()

    # Join the binary data into a single string
    content = ''.join(data)

    # Use StringIO to convert the string data to a file-like object
    file = StringIO(content)

    # Create a DataFrame
    df = pd.read_csv(file, sep='|')

    # Filter for active tickers
    active_tickers = df[df['Test Issue'] == 'N']

    return active_tickers['Symbol'].tolist()

# Retrieve NASDAQ tickers
nasdaq_tickers = get_nasdaq_tickers()

# Function to fetch news headlines from Yahoo Finance
def fetch_yahoo_finance_headlines():
    headlines = []
    url = "https://finance.yahoo.com/rss/topstories"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        today_date = datetime.datetime.now().date()
        
        for item in items:
            headline = item.title.text if item.title else ''
            link = item.link.text if item.link else ''
            pub_date = item.pubDate.text if item.pubDate else ''

            # Ensure pub_date is parsed correctly
            pub_date_parsed = None
            if pub_date:
                try:
                    pub_date_parsed = datetime.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        pub_date_parsed = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    except ValueError:
                        pub_date_parsed = None

            # Ensure headline is a string and not empty
            if isinstance(headline, str) and headline and pub_date_parsed:
                if (
                    (any(ticker in headline for ticker in nasdaq_tickers if isinstance(ticker, str)) or "NASDAQ" in headline.upper()) 
                    and pub_date_parsed.date() == today_date
                ):
                    headlines.append({"headline": headline, "link": link})
    except requests.RequestException as e:
        st.error(f"An error occurred while fetching Yahoo Finance news: {e}")

    return headlines

# Function to fetch news headlines from Google News
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
            headline = item.title.text if item.title else ''
            link = item.link.text if item.link else ''
            pub_date = item.pubDate.text if item.pubDate else ''

            # Ensure pub_date is parsed correctly
            pub_date_parsed = None
            if pub_date:
                try:
                    pub_date_parsed = datetime.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        pub_date_parsed = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    except ValueError:
                        pub_date_parsed = None

            # Ensure headline is a string and not empty
            if isinstance(headline, str) and headline and pub_date_parsed:
                if (
                    (any(isinstance(ticker, str) and ticker in headline for ticker in nasdaq_tickers) or "NASDAQ" in headline.upper()) 
                    and pub_date_parsed.date() == today_date
                ):
                    headlines.append({"headline": headline, "link": link})
    except requests.RequestException as e:
        st.error(f"An error occurred while fetching Google News: {e}")

    return headlines

# Function to fetch all news headlines
def fetch_news_headlines():
    yahoo_headlines = fetch_yahoo_finance_headlines()
    google_headlines = fetch_google_news_headlines()
    return yahoo_headlines + google_headlines

# Function to extract ticker symbols from headlines
def extract_ticker(headline):
    # Look for patterns like "(AAPL)" or "AAPL" in context
    match = re.search(r'\b([A-Z]{1,5})\b', headline)
    if match and match.group(1) in nasdaq_tickers:
        return match.group(1)
    return None

# Function to fetch article content
def fetch_article_content(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        article_text = ' '.join(p.get_text() for p in paragraphs)
        return article_text if article_text else None
    except requests.RequestException:
        return None

# Function to analyze sentiment
def analyze_sentiment(headlines):
    sentiments = []

    for item in headlines:
        headline = item['headline']
        link = item['link']
        article_text = fetch_article_content(link)
        if article_text is None:
            continue
        full_text = headline + " " + article_text
        score = sia.polarity_scores(full_text)
        ticker = extract_ticker(headline)
        if ticker:
            sentiments.append({
                'headline': headline,
                'link': link,
                'ticker': ticker,
                'sentiment_score': score['compound'],
                'sentiment_label': 'Positive' if score['compound'] > 0.05 else 'Negative' if score['compound'] < -0.05 else 'Neutral'
            })

    return sentiments

# Function to determine potential gainers
def identify_potential_gainers(sentiments, threshold=0.3):
    potential_gainers = [entry for entry in sentiments if entry['sentiment_score'] > threshold]
    return potential_gainers

# Function to generate word cloud from headlines
def generate_wordcloud(headlines):
    text = " ".join(headlines)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Function to perform technical analysis and predictions
def technical_analysis_and_prediction(tickers):
    if not tickers:
        st.write("No tickers found for technical analysis.")
        return []

    results = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period='3mo', interval='1d')
            if data.empty:
                continue

            # Calculate moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Calculate recent indicators
            latest_rsi = data['RSI'].iloc[-1]
            latest_macd = data['MACD'].iloc[-1]
            latest_signal = data['Signal'].iloc[-1]
            latest_close = data['Close'].iloc[-1]
            latest_sma20 = data['SMA_20'].iloc[-1]
            latest_sma50 = data['SMA_50'].iloc[-1]

            # Predictions based on technical indicators
            prediction = "Hold"
            if latest_rsi < 30 and latest_macd > latest_signal and latest_close > latest_sma20:
                prediction = "Strong Buy"
            elif latest_rsi > 70 and latest_macd < latest_signal:
                prediction = "Strong Sell"
            elif latest_sma20 > latest_sma50:
                prediction = "Buy"
            elif latest_sma20 < latest_sma50:
                prediction = "Sell"

            # Append result for each ticker
            results.append({
                'Ticker': ticker,
                'RSI': latest_rsi,
                'MACD': latest_macd,
                'Signal': latest_signal,
                'SMA_20': latest_sma20,
                'SMA_50': latest_sma50,
                'Prediction': prediction
            })
        except Exception as e:
            st.write(f"An error occurred while processing {ticker}: {e}")

    return results

# Function to identify chart patterns
def identify_chart_patterns(data):
    # Placeholder for pattern recognition logic
    # Here we would include pattern recognition algorithms to detect common patterns
    patterns = []
    closing_prices = data['Close'].tolist()

    # Dummy pattern detection (Replace this with actual pattern detection logic)
    if closing_prices[-1] > closing_prices[-2] > closing_prices[-3]:  # Example pattern
        patterns.append("Uptrend")

    return patterns

# Function to display results in a DataFrame
def display_results(potential_gainers, top_technical_stocks):
    if not potential_gainers:
        st.write("No potential gainers found today.")
        return

    df = pd.DataFrame(potential_gainers)
    df.sort_values(by='sentiment_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    st.write("### Potential Gainers")
    st.dataframe(df[['ticker', 'headline', 'link', 'sentiment_score', 'sentiment_label']])

    # Generate and display word cloud for positive headlines
    positive_headlines = [headline for headline in df['headline'].tolist() if headline is not None]
    st.write("### Word Cloud of Positive Headlines")
    generate_wordcloud(positive_headlines)

    # Display stocks from technical analysis
    if top_technical_stocks:
        st.write("### Stocks Based on Technical Analysis")
        tech_df = pd.DataFrame(top_technical_stocks, columns=['Ticker', 'RSI', 'MACD', 'Signal', 'SMA_20', 'SMA_50', 'Prediction'])
        st.dataframe(tech_df)

# Function to plot candlestick charts and identify patterns
def plot_candlestick_chart(ticker, interval, chart_type):
    data = yf.download(ticker, period='5d', interval=interval)
    if data.empty:
        st.write(f"No data available for {ticker} with interval {interval}.")
        return

    fig, axlist = mpf.plot(
        data,
        type=chart_type,
        style='yahoo',
        title=f"{ticker} - {interval}",
        volume=True,
        returnfig=True
    )

    st.pyplot(fig)

    # Identify chart patterns
    patterns = identify_chart_patterns(data)
    if patterns:
        st.write(f"Identified Patterns for {ticker}: {', '.join(patterns)}")
        # Provide prediction based on patterns
        for pattern in patterns:
            if pattern == "Uptrend":
                st.write("Prediction: The stock may continue to rise.")
            # Add more predictions based on different patterns here
    else:
        st.write(f"No significant patterns identified for {ticker}.")

# Function for sentiment analysis of news articles
def sentiment_analysis(text):
    return sia.polarity_scores(text)

# Function to fetch and analyze news articles related to a specific stock
def analyze_news_articles(stock_ticker):
    st.header("Finance News Analysis")
    st.subheader("Finance News Articles")
    query = f"{stock_ticker} stock OR share"
    url = f"https://news.google.com/rss/search?q={query}"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")

    articles = []
    today_date = datetime.datetime.now().date()
    for item in items:
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        pub_date_parsed = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
        if pub_date_parsed.date() == today_date:
            articles.append({"title": title, "link": link, "pub_date": pub_date})

    news_df = pd.DataFrame(articles)

    # Highlight positive and negative words
    def highlight_sentiment_words(text):
        words = text.split()
        highlighted_text = []
        for word in words:
            score = sia.polarity_scores(word)
            if score['compound'] >= 0.1:
                highlighted_text.append(f'<span style="background-color: lightgreen">{word}</span>')
            elif score['compound'] <= -0.1:
                highlighted_text.append(f'<span style="background-color: lightcoral">{word}</span>')
            else:
                highlighted_text.append(word)
        return ' '.join(highlighted_text)

    if not news_df.empty:
        news_df["highlighted_title"] = news_df["title"].apply(highlight_sentiment_words)
        news_df["link"] = news_df["link"].apply(lambda x: f'<a href="{x}" target="_blank">LINK</a>')

        # Create a dropdown to display all articles
        st.subheader("News Articles")
        for index, row in news_df.iterrows():
            st.markdown(f"{index + 1}. {row['highlighted_title']} [LINK]({row['link']})", unsafe_allow_html=True)

        # Display sentiment summary
        news_df["sentiment"] = news_df["title"].apply(lambda x: "positive" if sia.polarity_scores(x)["compound"] >= 0.05 else "negative" if sia.polarity_scores(x)["compound"] <= -0.05 else "neutral")
        sentiment_summary = news_df["sentiment"].value_counts().reset_index()
        sentiment_summary.columns = ["Sentiment", "Count"]
        st.subheader("Sentiment Summary")
        st.table(sentiment_summary)
        
        # Display word cloud
        st.subheader("Word Cloud of News Titles")
        text = " ".join(news_df["title"].tolist())
        wordcloud = WordCloud(stopwords=nltk.corpus.stopwords.words("english")).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No news articles found.")

def fetch_and_identify_stocks():
    """Function to fetch news and identify potential stocks."""
    st.header("News Fetch and Analysis")
    st.write("This step fetches the latest headlines and identifies NASDAQ stocks based on sentiment analysis.")
    
    with st.spinner('Fetching and analyzing news...'):
        headlines = fetch_news_headlines()
        if not headlines:
            return

        with st.expander("Show/Hide Fetched Headlines", expanded=True):
            st.write("### Fetched Headlines")
            for headline in headlines:
                st.write(headline['headline'])

        # Extract tickers from headlines
        tickers_in_news = list({extract_ticker(item['headline']) for item in headlines if extract_ticker(item['headline'])})

        # Display all tickers mentioned in the news
        st.write("### Tickers Mentioned in News")
        st.write(", ".join(tickers_in_news))

        # Analyze sentiment of each headline
        sentiments = analyze_sentiment(headlines)
        st.write("### Headline Sentiments")
        for entry in sentiments:
            st.write(f"{entry['headline']} - Sentiment Score: {entry['sentiment_score']} ({entry['sentiment_label']})")

        # Identify potential gainers
        gainers = identify_potential_gainers(sentiments)

        # Perform technical analysis
        top_technical_stocks = technical_analysis_and_prediction(tickers_in_news)

        # Display results
        display_results(gainers, top_technical_stocks)

        # Store top tickers for analysis
        st.session_state['tickers'] = tickers_in_news

def analysis_page():
    """Function to select analysis type and perform analysis."""
    st.header("Stock Analysis")
    st.write("Select which stocks to analyze and the type of analysis to perform.")

    # Check if tickers are available
    if 'tickers' not in st.session_state or not st.session_state['tickers']:
        st.warning("Please fetch and identify stocks first.")
        return

    selected_tickers = st.multiselect(
        "Select NASDAQ Tickers for Analysis",
        options=st.session_state['tickers']
    )

    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=["Technical Analysis and Prediction", "News Sentiment Analysis"]
    )

    # Sidebar options for chart type and interval
    st.sidebar.title("Chart Settings")
    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        options=["candle", "line", "ohlc"]
    ).capitalize()

    intervals = ["1m", "2m", "5m", "15m", "30m", "1h", "2h", "1d"]
    selected_interval = st.sidebar.selectbox("Select Time Interval for Chart", intervals)

    # Automatically display analysis results when selections are made
    if analysis_type == "Technical Analysis and Prediction":
        for ticker in selected_tickers:
            st.write(f"### {ticker} - {chart_type} Chart")
            plot_candlestick_chart(ticker, selected_interval, chart_type)

        analysis_results = technical_analysis_and_prediction(selected_tickers)
        st.write("### Selected Stocks - Technical Analysis and Prediction")
        st.dataframe(pd.DataFrame(analysis_results))

    elif analysis_type == "News Sentiment Analysis":
        st.write("### Selected Stocks - News Sentiment Analysis")
        for ticker in selected_tickers:
            analyze_news_articles(ticker)

# Function to display a glossary in the sidebar
def glossary():
    """Function to add a glossary to the sidebar."""
    st.sidebar.title("Glossary")
    
    glossary_terms = {
        "NASDAQ": "The National Association of Securities Dealers Automated Quotations, an American stock exchange.",
        "Ticker": "A unique series of letters representing a particular publicly traded company.",
        "Sentiment Analysis": "A process of determining the sentiment expressed in a piece of text, such as positive, negative, or neutral.",
        "VADER": "Valence Aware Dictionary and sEntiment Reasoner, a sentiment analysis tool designed to detect sentiment in social media text.",
        "SMA (Simple Moving Average)": "A calculation that takes the arithmetic mean of a given set of prices over a specified number of days.",
        "RSI (Relative Strength Index)": "A momentum oscillator that measures the speed and change of price movements.",
        "MACD (Moving Average Convergence Divergence)": "A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
        "Candlestick Chart": "A type of financial chart used to describe price movements of a security, derivative, or currency.",
        "Word Cloud": "A visual representation of text data, showing the frequency of words in varying sizes.",
    }
    
    for term, definition in glossary_terms.items():
        st.sidebar.markdown(f"**{term}**: {definition}")

# Main function to navigate between pages
def main():
    # Initialize session state for app page
    if 'app_page' not in st.session_state:
        st.session_state['app_page'] = "Home"

    # Initialize session state for tickers
    if 'tickers' not in st.session_state:
        st.session_state['tickers'] = []

    # Page navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analysis"])

    # Call the glossary function
    glossary()

    # Show the appropriate page
    if page == "Home":
        fetch_and_identify_stocks()
    elif page == "Analysis":
        analysis_page()

if __name__ == "__main__":
    main()
