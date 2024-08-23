import streamlit as st
import pandas as pd
import yfinance as yf
import nltk
import pandas_ta as ta
import mplfinance as mpf
from datetime import datetime
from ftplib import FTP
from io import StringIO
import requests
from bs4 import BeautifulSoup

# Ensure NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

        all_tickers = pd.concat([nasdaq_df, other_df], ignore_index=True)
        ftp.quit()
        return all_tickers

    except Exception as e:
        st.error(f"An error occurred while fetching tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Security Name', 'Exchange'])

# Function to scrape Yahoo Finance stock market news
def scrape_yahoo_finance_news():
    url = "https://finance.yahoo.com/topic/stock-market-news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for item in soup.find_all('li', class_='js-stream-content')[:5]:  # Limiting to 5 articles
        title = item.find('h3').get_text(strip=True)
        link = "https://finance.yahoo.com" + item.find('a')['href']
        summary = item.find('p').get_text(strip=True) if item.find('p') else "No summary available"
        articles.append({
            'title': title,
            'url': link,
            'summary': summary
        })

    return articles

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

# Function to fetch intraday data
def fetch_intraday_data(ticker, interval="5m"):
    data = yf.download(tickers=ticker, period="1d", interval=interval)
    return data

# Function to apply risk management and filter positions for today's date
def apply_risk_management(stock, buy_signals, sell_signals, data):
    positions = []
    stop_loss = 0.95  # Example: 5% below buy price
    take_profit = 1.10  # Example: 10% above buy price

    for i in range(len(data)):
        if buy_signals.iloc[i]:
            buy_price = data['Close'].iloc[i]
            positions.append(('Buy', buy_price, data.index[i]))
        elif sell_signals.iloc[i]:
            if len(positions) > 0 and positions[-1][0] == 'Buy':
                sell_price = data['Close'].iloc[i]
                positions.append(('Sell', sell_price, data.index[i]))
        elif len(positions) > 0 and positions[-1][0] == 'Buy':
            current_price = data['Close'].iloc[i]
            if current_price <= positions[-1][1] * stop_loss:
                positions.append(('Sell (Stop Loss)', current_price, data.index[i]))
            elif current_price >= positions[-1][1] * take_profit:
                positions.append(('Sell (Take Profit)', current_price, data.index[i]))

    # Filter positions based on today's date
    today = datetime.now().date()
    filtered_positions = [pos for pos in positions if pos[2].date() == today]

    return filtered_positions

# Function to plot data using candlestick chart with 5-minute intervals and moving averages
def plot_intraday_data_with_moving_averages(data, stock, buy_signals, sell_signals):
    if data.empty:
        st.write("No data available for today.")
        return

    # Calculate 20-period and 50-period moving averages
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['50_SMA'] = data['Close'].rolling(window=50).mean()

    # Plot the candlestick chart with moving averages
    st.write("## 5-Minute Interval Candlestick Chart with Moving Averages for Today")
    fig, ax = mpf.plot(data, type='candle', volume=True, 
                       title=f'{stock} - 5-Minute Interval Candlestick Chart for Today',
                       style='yahoo', figsize=(10, 6), tight_layout=True,
                       addplot=[
                           mpf.make_addplot(data['20_SMA'], color='blue', width=0.75),
                           mpf.make_addplot(data['50_SMA'], color='orange', width=0.75),
                           mpf.make_addplot(buy_signals, type='scatter', marker='^', color='green'),
                           mpf.make_addplot(sell_signals, type='scatter', marker='v', color='red')
                       ], returnfig=True)
    st.pyplot(fig)

# Main function for the Streamlit app
def main():
    st.title("Stock Analysis Dashboard")

    all_tickers_df = get_us_stock_tickers()
    if all_tickers_df.empty:
        st.error("Failed to load US stock ticker data.")
        return

    ticker_set = set(all_tickers_df['Symbol'])

    # Scrape Yahoo Finance stock market news
    news_articles = scrape_yahoo_finance_news()

    if not news_articles:
        st.error("Failed to scrape news articles.")
        return
    
    st.subheader("5 Most Recent Yahoo Finance Stock Market News Articles")
    for article in news_articles:  # Display the 5 most recent articles
        st.write(f"**{article['title']}**")
        st.write(f"{article['summary']}")
        st.write(f"[Read more]({article['url']})")
        st.write("---")

    selected_ticker = st.selectbox("Select Ticker for Analysis", list(ticker_set)[:10])

    if not selected_ticker:
        st.error("No ticker selected.")
        return

    company_profile = fetch_company_info(selected_ticker)
    if company_profile:
        st.header(f"Company Profile for {company_profile['Name']}")
        st.write(f"**Sector:** {company_profile['Sector']}")
        st.write(f"**Industry:** {company_profile['Industry']}")
        st.write(f"**Market Cap:** {company_profile['Market Cap']}")
        st.write(f"**Full Time Employees:** {company_profile['Full Time Employees']}")
        st.write(f"**Description:** {company_profile['Description']}")

    # Fetch intraday data for the selected ticker
    intraday_data = fetch_intraday_data(selected_ticker)

    # Generate buy and sell signals using RSI
    buy_signals = intraday_data['RSI'] < 30
    sell_signals = intraday_data['RSI'] > 70

    st.header("Technical Analysis")
    
    # Apply risk management to generate positions
    positions = apply_risk_management(selected_ticker, buy_signals, sell_signals, intraday_data)
    
    st.subheader("Risk Management Positions for Today")
    if positions:
        for position in positions:
            st.write(f"{position[0]} at {position[1]:.2f} on {position[2]}")
    else:
        st.write("No positions triggered today.")

    # Plot the intraday data with moving averages and signals
    plot_intraday_data_with_moving_averages(intraday_data, selected_ticker, buy_signals, sell_signals)

if __name__ == "__main__":
    main()
