import yfinance as yf
import pandas as pd
import pandas_ta as ta
import talib
import mplfinance as mpf
import streamlit as st

def fetch_stock_data(ticker, period='1mo', interval='1d'):
    try:
        # Fetch stock data using yfinance
        stock_data = yf.download(ticker, period=period, interval=interval)
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        return stock_data
    except Exception as e:
        st.error(f"Failed to download data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df):
    # Add technical indicators
    df['SMA'] = ta.sma(df['Close'], length=20)
    df['EMA'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    bollinger = ta.bbands(df['Close'], length=20)
    df['Bollinger_Mid'] = bollinger['BBM_20_2.0']
    df['Bollinger_Upper'] = bollinger['BBU_20_2.0']
    df['Bollinger_Lower'] = bollinger['BBL_20_2.0']
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    # Additional indicators from TA-Lib
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df

def identify_candlestick_patterns(df):
    # Identify candlestick patterns using TA-Lib
    patterns = {
        'Bullish Hammer': talib.CDLHAMMER,
        'Shooting Star': talib.CDLSHOOTINGSTAR,
        'Bullish Engulfing': talib.CDLENGULFING,
        'Bearish Engulfing': talib.CDLENGULFING,
        'Morning Star': talib.CDLMORNINGSTAR,
        'Evening Star': talib.CDLEVENINGSTAR
    }
    
    pattern_results = {name: func(df['Open'], df['High'], df['Low'], df['Close']) for name, func in patterns.items()}
    
    for pattern_name, results in pattern_results.items():
        df[pattern_name] = results.apply(lambda x: pattern_name if x != 0 else None)
        
    return df

def plot_stock_data(df, ticker):
    # Plot the stock data with mplfinance
    fig, axes = mpf.plot(
        df,
        type='candle',
        addplot=[
            mpf.make_addplot(df['SMA'], color='green', width=0.75),
            mpf.make_addplot(df['EMA'], color='blue', width=0.75),
            mpf.make_addplot(df['Bollinger_Upper'], color='orange', width=0.75),
            mpf.make_addplot(df['Bollinger_Lower'], color='orange', width=0.75),
            mpf.make_addplot(df['RSI'], panel=1, color='purple', secondary_y=True),
            mpf.make_addplot(df['MACD'], panel=2, color='red', secondary_y=True),
            mpf.make_addplot(df['MACD_Signal'], panel=2, color='blue', secondary_y=True),
            mpf.make_addplot(df['OBV'], panel=3, color='brown', secondary_y=True),
        ],
        volume=True,
        title=f'{ticker} Stock Price and Indicators',
        returnfig=True
    )
    return fig

def explain_indicators():
    st.sidebar.header("Indicator Explanations")
    st.sidebar.markdown("""
    **SMA (Simple Moving Average):** An average of closing prices over a specified period. Used to identify trends.
    
    **EMA (Exponential Moving Average):** Similar to SMA but gives more weight to recent prices. Reacts faster to price changes.
    
    **RSI (Relative Strength Index):** Measures the speed and change of price movements on a scale of 0 to 100. An RSI above 70 indicates overbought conditions, and below 30 indicates oversold conditions.
    
    **MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator that shows the relationship between two moving averages. A bullish signal is generated when the MACD line crosses above the signal line, and a bearish signal is generated when it crosses below.
    
    **Bollinger Bands:** Consists of a middle band (SMA) and two outer bands (standard deviations above and below the SMA). The bands expand and contract based on market volatility.
    
    **OBV (On-Balance Volume):** Measures buying and selling pressure. It adds volume on up days and subtracts volume on down days.

    **ADX (Average Directional Index):** Measures trend strength. A value above 25 indicates a strong trend.

    **CCI (Commodity Channel Index):** Identifies cyclical trends in a market. A value above 100 may indicate an overbought condition, and below -100 may indicate an oversold condition.

    **ATR (Average True Range):** Measures market volatility. High ATR values indicate high volatility, while low values indicate low volatility.
    """)

def make_predictions(df):
    st.header("Predictions")
    st.write("The following are potential predictions based on the latest data and indicators:")
    
    # Example of predictions based on indicators
    if df['RSI'].iloc[-1] > 70:
        st.write("The stock might be overbought, indicating a potential price decline.")
    elif df['RSI'].iloc[-1] < 30:
        st.write("The stock might be oversold, indicating a potential price increase.")
    
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        st.write("MACD is bullish: Potential upward momentum.")
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
        st.write("MACD is bearish: Potential downward momentum.")
    
    # Adding additional checks for candlestick patterns
    if 'Bullish Hammer' in df.columns and df['Bullish Hammer'].iloc[-1] == 'Bullish Hammer':
        st.write("Bullish Hammer detected: Possible upward reversal.")
    if 'Shooting Star' in df.columns and df['Shooting Star'].iloc[-1] == 'Shooting Star':
        st.write("Shooting Star detected: Possible downward reversal.")
    
    # Example prediction logic for ADX
    if df['ADX'].iloc[-1] > 25:
        st.write("Strong trend detected according to ADX.")

def main():
    st.title("Real-Time Stock Analysis App")

    # Input stock ticker symbol
    ticker = st.text_input("Enter the stock ticker:", "AAPL")

    # Select interval
    interval = st.selectbox("Select interval:", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])

    # Select period
    period = st.selectbox("Select period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])

    if ticker:
        # Fetch data
        stock_data = fetch_stock_data(ticker, period=period, interval=interval)

        if not stock_data.empty:
            # Calculate indicators
            stock_data = calculate_indicators(stock_data)

            # Identify candlestick patterns
            stock_data = identify_candlestick_patterns(stock_data)

            # Display patterns identified
            st.write(stock_data[['Bullish Hammer', 'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing', 'Morning Star', 'Evening Star']].dropna(how='all'))

            # Plot data
            fig = plot_stock_data(stock_data, ticker)
            st.pyplot(fig)

            # Explain indicators
            explain_indicators()

            # Make predictions
            make_predictions(stock_data)

if __name__ == "__main__":
    main()
