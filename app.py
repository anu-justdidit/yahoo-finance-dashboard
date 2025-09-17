import dash
from dash import html, dcc, Input, Output, callback, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, title="Advanced Financial Dashboard", update_title="Loading...")
server = app.server

# ======================
# CONFIGURATION
# ======================
class Config:
    PORT = int(os.environ.get("PORT", 8050))
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    CACHE_TIMEOUT = 300  # 5 minutes
    NEWS_UPDATE_INTERVAL = 300000  # 5 minutes
    MAX_NEWS_ARTICLES = 5
    TECHNICAL_INDICATORS = {
        'rsi': {'window': 14, 'name': 'RSI (14)'},
        'sma': {'windows': [20, 50], 'name': 'Simple Moving Average'},
        'ema': {'windows': [20, 50], 'name': 'Exponential Moving Average'},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9, 'name': 'MACD'},
        'bollinger': {'window': 20, 'num_std': 2, 'name': 'Bollinger Bands'}
    }
    STOCKS = [
        {'label': 'S&P 500 (^GSPC)', 'value': '^GSPC'},
        {'label': 'Apple (AAPL)', 'value': 'AAPL'},
        {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
        {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
        {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
        {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
        {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
        {'label': 'Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
        {'label': 'Ethereum (ETH-USD)', 'value': 'ETH-USD'}
    ]
    TIME_PERIODS = [
        {'label': '1 Month', 'value': '1mo'},
        {'label': '3 Months', 'value': '3mo'},
        {'label': '6 Months', 'value': '6mo'},
        {'label': '1 Year', 'value': '1y'},
        {'label': '2 Years', 'value': '2y'},
        {'label': '5 Years', 'value': '5y'}
    ]

# ======================
# CACHING DECORATOR
# ======================
def cache_decorator(timeout):
    def decorator(func):
        cache = {}
        def wrapper(*args):
            current_time = time.time()
            key = str(args)
            if key in cache and current_time - cache[key]['time'] < timeout:
                return cache[key]['value']
            result = func(*args)
            cache[key] = {'value': result, 'time': current_time}
            return result
        return wrapper
    return decorator

# ======================
# TECHNICAL INDICATOR FUNCTIONS
# ======================
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window=20):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=20):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band, sma

def calculate_technical_indicators(data, indicators):
    """Calculate all selected technical indicators"""
    results = {}
    
    if 'rsi' in indicators:
        results['rsi'] = calculate_rsi(data, Config.TECHNICAL_INDICATORS['rsi']['window'])
    
    if 'sma' in indicators:
        for window in Config.TECHNICAL_INDICATORS['sma']['windows']:
            results[f'sma_{window}'] = calculate_sma(data, window)
    
    if 'ema' in indicators:
        for window in Config.TECHNICAL_INDICATORS['ema']['windows']:
            results[f'ema_{window}'] = calculate_ema(data, window)
    
    if 'macd' in indicators:
        macd, signal, histogram = calculate_macd(
            data, 
            Config.TECHNICAL_INDICATORS['macd']['fast'],
            Config.TECHNICAL_INDICATORS['macd']['slow'],
            Config.TECHNICAL_INDICATORS['macd']['signal']
        )
        results['macd'] = macd
        results['macd_signal'] = signal
        results['macd_histogram'] = histogram
    
    if 'bollinger' in indicators:
        upper, lower, middle = calculate_bollinger_bands(
            data,
            Config.TECHNICAL_INDICATORS['bollinger']['window'],
            Config.TECHNICAL_INDICATORS['bollinger']['num_std']
        )
        results['bollinger_upper'] = upper
        results['bollinger_lower'] = lower
        results['bollinger_middle'] = middle
    
    return results

# ======================
# DATA FETCHING FUNCTIONS
# ======================
@cache_decorator(timeout=Config.CACHE_TIMEOUT)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance with retry logic and error handling"""
    try:
        for attempt in range(3):
            try:
                logger.info(f"Fetching data for {ticker}, period: {period}, attempt {attempt + 1}")
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if hist.empty:
                    logger.warning(f"No data for {ticker}, attempt {attempt + 1}")
                    time.sleep(1)
                    continue
                    
                # Add additional metrics
                hist['Daily_Return'] = hist['Close'].pct_change()
                hist['Cumulative_Return'] = (1 + hist['Daily_Return']).cumprod() - 1
                
                return hist
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                time.sleep(2)
                
        return None
        
    except Exception as e:
        logger.error(f"Critical error with {ticker}: {e}")
        return None

def load_data_from_csv(ticker):
    """Load data from CSV file if available"""
    try:
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            logger.info(f"Loaded data from CSV for {ticker}")
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading CSV for {ticker}: {e}")
        return None

def get_stock_data(ticker, period="1y"):
    """Get stock data with multiple fallback strategies"""
    # First try to load from CSV
    csv_data = load_data_from_csv(ticker)
    if csv_data is not None:
        # Filter data based on the requested period
        if period == "1mo":
            return csv_data.last('30D')
        elif period == "3mo":
            return csv_data.last('90D')
        elif period == "6mo":
            return csv_data.last('180D')
        elif period == "1y":
            return csv_data.last('365D')
        elif period == "2y":
            return csv_data.last('730D')
        elif period == "5y":
            return csv_data.last('1825D')
        return csv_data
    
    # If CSV not available, try API
    real_data = fetch_stock_data(ticker, period)
    
    if real_data is not None and not real_data.empty:
        return real_data
    
    # Fallback to mock data
    logger.warning(f"Using mock data for {ticker}")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(200, 300, 100),
        'Low': np.random.uniform(50, 100, 100),
        'Close': np.random.uniform(150, 250, 100),
        'Volume': np.random.randint(1000000, 5000000, 100),
        'Daily_Return': np.random.normal(0.001, 0.02, 100),
    }, index=dates)
    mock_data['Cumulative_Return'] = (1 + mock_data['Daily_Return']).cumprod() - 1
    
    return mock_data

# ======================
# NEWS FETCHING FUNCTION
# ======================
@cache_decorator(timeout=Config.CACHE_TIMEOUT)
def fetch_financial_news(ticker="AAPL", max_articles=5):
    """Fetch financial news for a given stock ticker"""
    try:
        # Using Yahoo Finance news RSS feed
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        
        # Fetch the RSS feed
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        
        # Extract news items
        news_items = []
        for item in soup.find_all('item')[:max_articles]:
            title = item.title.text if item.title else "No title"
            link = item.link.text if item.link else "#"
            pub_date = item.pubDate.text if item.pubDate else "No date"
            
            # Extract description/summary if available
            description = ""
            if item.description:
                description = item.description.text
            elif item.content_encoded:
                description = item.content_encoded.text
            
            # Clean up description (remove HTML tags)
            description = BeautifulSoup(description, "html.parser").get_text()[:200] + "..."
            
            news_items.append({
                'title': title,
                'link': link,
                'pub_date': pub_date,
                'description': description
            })
        
        logger.info(f"Fetched {len(news_items)} news articles for {ticker}")
        return news_items
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        # Return mock news data if API fails
        return [
            {
                'title': f'Market Update: {ticker} Shows Strong Performance',
                'link': '#',
                'pub_date': datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'),
                'description': f'{ticker} stock continues to show strength in today\'s trading session, outperforming market expectations.'
            },
            {
                'title': f'Analysts Raise Price Target for {ticker}',
                'link': '#',
                'pub_date': (datetime.now() - timedelta(hours=2)).strftime('%a, %d %b %Y %H:%M:%S GMT'),
                'description': f'Leading analysts have increased their price targets for {ticker} following strong quarterly earnings.'
            }
        ]

# ======================
# APP LAYOUT
# ======================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📈 Advanced Financial Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Real-time market data with technical analysis", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    # Controls Section
    html.Div([
        # Stock Selector
        html.Div([
            html.Label("Select Stock:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='stock-selector',
                options=Config.STOCKS,
                value='AAPL',
                clearable=False,
                style={'width': '220px'}
            ),
        ], style={'marginRight': '20px'}),
        
        # Time Period Selector
        html.Div([
            html.Label("Time Period:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='date-range',
                options=Config.TIME_PERIODS,
                value='1y',
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'marginRight': '20px'}),
        
        # Technical Indicator Selector
        html.Div([
            html.Label("Technical Indicators:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='indicator-selector',
                options=[
                    {'label': 'RSI (Relative Strength Index)', 'value': 'rsi'},
                    {'label': 'SMA (Simple Moving Average)', 'value': 'sma'},
                    {'label': 'EMA (Exponential Moving Average)', 'value': 'ema'},
                    {'label': 'MACD', 'value': 'macd'},
                    {'label': 'Bollinger Bands', 'value': 'bollinger'}
                ],
                value=['sma', 'rsi'],
                multi=True,
                clearable=False,
                style={'width': '300px'}
            ),
        ]),
    ], style={
        'padding': '20px', 
        'display': 'flex', 
        'alignItems': 'center', 
        'justifyContent': 'center', 
        'flexWrap': 'wrap',
        'backgroundColor': 'white',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'margin': '10px 20px',
        'borderRadius': '5px'
    }),
    
    # Error message div
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'padding': '10px'}),
    
    # Current price metrics
    html.Div(id='stock-metrics', style={
        'padding': '15px', 
        'textAlign': 'center', 
        'backgroundColor': '#f8f9fa', 
        'borderRadius': '5px',
        'margin': '10px 20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Main content area
    html.Div([
        # Left column - Chart
        html.Div([
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=dcc.Graph(
                    id='stock-graph', 
                    style={'height': '65vh'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            )
        ], style={'width': '70%', 'padding': '10px'}),
        
        # Right column - Additional info
        html.Div([
            html.H3("Performance Metrics", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div(id='performance-metrics'),
            
            html.H3("Volume", style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'marginTop': '20px'}),
            dcc.Graph(id='volume-chart', style={'height': '200px'}),
        ], style={'width': '30%', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'})
    ], style={'display': 'flex', 'margin': '10px 20px'}),
    
    # News section
    html.Div([
        html.H2("📰 Latest Financial News", 
                style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
        html.Div(id='financial-news')
    ], style={'padding': '15px', 'margin': '10px 20px'}),
    
    # Footer
    html.Div([
        html.P(f"© {datetime.now().year} Financial Dashboard | Data provided by Yahoo Finance", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '0'}),
        html.P("Last updated: ", id='last-updated', 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '5px 0'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginTop': '30px'}),
    
    # Interval components
    dcc.Interval(
        id='data-interval',
        interval=60000,  # Update data every 1 minute
        n_intervals=0
    ),
    dcc.Interval(
        id='news-interval',
        interval=Config.NEWS_UPDATE_INTERVAL,
        n_intervals=0
    ),
    dcc.Store(id='data-store')  # Store for caching data
])

# ======================
# CALLBACKS
# ======================
@app.callback(
    [Output('data-store', 'data'),
     Output('error-message', 'children'),
     Output('last-updated', 'children')],
    [Input('stock-selector', 'value'),
     Input('date-range', 'value'),
     Input('data-interval', 'n_intervals')]
)
def update_stored_data(selected_stock, date_range, n_intervals):
    """Fetch and store data with error handling"""
    if not selected_stock:
        return {}, "", "Last updated: Never"
    
    try:
        data = get_stock_data(selected_stock, date_range)
        
        if data is None or data.empty:
            error_msg = "⚠️ Unable to fetch data. Please try another stock or check your internet connection."
            return {}, error_msg, "Last updated: Error"
        
        # Convert to JSON-serializable format
        data_json = {
            'index': data.index.strftime('%Y-%m-%d').tolist(),
            'Open': data['Open'].fillna(0).tolist(),
            'High': data['High'].fillna(0).tolist(),
            'Low': data['Low'].fillna(0).tolist(),
            'Close': data['Close'].fillna(0).tolist(),
            'Volume': data['Volume'].fillna(0).tolist(),
            'Daily_Return': data['Daily_Return'].fillna(0).tolist(),
            'Cumulative_Return': data['Cumulative_Return'].fillna(0).tolist()
        }
        
        timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return data_json, "", timestamp
        
    except Exception as e:
        logger.error(f"Error in update_stored_data: {e}")
        error_msg = "⚠️ An error occurred while fetching data. Please try again later."
        return {}, error_msg, "Last updated: Error"

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('stock-metrics', 'children'),
     Output('performance-metrics', 'children'),
     Output('volume-chart', 'figure'),
     Output('financial-news', 'children')],
    [Input('data-store', 'data'),
     Input('indicator-selector', 'value'),
     Input('stock-selector', 'value'),
     Input('news-interval', 'n_intervals')]
)
def update_dashboard(data_json, selected_indicators, selected_stock, n_intervals):
    """Update all dashboard components"""
    # Initialize empty figures and components
    fig = go.Figure()
    metrics = html.Div()
    performance_metrics = html.Div()
    volume_fig = go.Figure()
    news_section = html.Div()
    
    if not data_json or not selected_stock:
        return fig, metrics, performance_metrics, volume_fig, news_section
    
    try:
        # Reconstruct DataFrame from stored data
        data = pd.DataFrame({
            'Open': data_json['Open'],
            'High': data_json['High'],
            'Low': data_json['Low'],
            'Close': data_json['Close'],
            'Volume': data_json['Volume'],
            'Daily_Return': data_json['Daily_Return'],
            'Cumulative_Return': data_json['Cumulative_Return']
        }, index=pd.to_datetime(data_json['index']))
        
        # Create subplots with secondary y-axis for indicators
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{selected_stock} Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=selected_stock,
            increasing_line_color='#2ecc71',
            decreasing_line_color='#e74c3c'
        ), row=1, col=1)
        
        # Calculate and add technical indicators
        if selected_indicators:
            indicators = calculate_technical_indicators(data, selected_indicators)
            
            # RSI (plotted on secondary y-axis)
            if 'rsi' in indicators:
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['rsi'], 
                    name='RSI (14)',
                    line=dict(color='purple', width=1.5),
                    yaxis='y2'
                ), row=1, col=1)
            
            # SMAs
            for key in indicators:
                if key.startswith('sma_'):
                    window = key.split('_')[1]
                    fig.add_trace(go.Scatter(
                        x=data.index, y=indicators[key], 
                        name=f'SMA {window}',
                        line=dict(width=1.5)
                    ), row=1, col=1)
            
            # EMAs
            for key in indicators:
                if key.startswith('ema_'):
                    window = key.split('_')[1]
                    fig.add_trace(go.Scatter(
                        x=data.index, y=indicators[key], 
                        name=f'EMA {window}',
                        line=dict(width=1.5, dash='dot')
                    ), row=1, col=1)
            
            # Bollinger Bands
            if 'bollinger_upper' in indicators:
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['bollinger_upper'], 
                    name='Upper Band',
                    line=dict(color='rgba(200, 200, 200, 0.7)', width=1),
                    fill=None
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=indicators['bollinger_lower'], 
                    name='Lower Band',
                    line=dict(color='rgba(200, 200, 200, 0.7)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.1)'
                ), row=1, col=1)
        
        # Add volume bar chart
        colors = ['#e74c3c' if data['Close'].iloc[i] < data['Open'].iloc[i] else '#2ecc71' 
                 for i in range(len(data))]
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{selected_stock} Stock Analysis',
            template='plotly_white',
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Add secondary y-axis for RSI if needed
        if selected_indicators and 'rsi' in selected_indicators:
            fig.update_layout(
                yaxis2=dict(
                    title="RSI",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    showgrid=False
                )
            )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Calculate metrics
        if len(data) > 0:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - prev_price
            percent_change = (price_change / prev_price) * 100
            
            # Calculate additional metrics
            period_high = data['High'].max()
            period_low = data['Low'].min()
            avg_volume = data['Volume'].mean()
            volatility = data['Daily_Return'].std() * np.sqrt(252)  # Annualized
            
            metrics = html.Div([
                html.H2(f"${current_price:.2f}", style={
                    'margin': '0', 
                    'color': '#2c3e50',
                    'fontSize': '2.5rem'
                }),
                html.P(f"{price_change:+.2f} ({percent_change:+.2f}%)", style={
                    'margin': '5px 0', 
                    'color': '#2ecc71' if price_change >= 0 else '#e74c3c',
                    'fontSize': '1.2rem',
                    'fontWeight': 'bold'
                }),
                html.Small(f"{selected_stock} • {data.index[-1].strftime('%Y-%m-%d %H:%M')}", 
                          style={'color': '#7f8c8d'})
            ])
            
            # Performance metrics table
            performance_metrics = html.Div([
                html.Div([
                    html.Span("Period High:", style={'fontWeight': 'bold'}),
                    html.Span(f" ${period_high:.2f}")
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Period Low:", style={'fontWeight': 'bold'}),
                    html.Span(f" ${period_low:.2f}")
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Avg Volume:", style={'fontWeight': 'bold'}),
                    html.Span(f" {avg_volume:,.0f}")
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Volatility:", style={'fontWeight': 'bold'}),
                    html.Span(f" {volatility:.2%}")
                ])
            ])
        
        # Create volume chart
        volume_fig = go.Figure(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            opacity=0.7,
            name='Volume'
        ))
        
        volume_fig.update_layout(
            template='plotly_white',
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=200
        )
        
        volume_fig.update_yaxes(title_text="Volume")
        
        # Fetch financial news
        news_items = fetch_financial_news(selected_stock, Config.MAX_NEWS_ARTICLES)
        
        # Create news cards
        news_cards = []
        for item in news_items:
            news_card = html.Div([
                html.H4(
                    html.A(item['title'], href=item['link'], target="_blank", 
                          style={'color': '#2c3e50', 'textDecoration': 'none', 'fontSize': '1.1rem'})
                ),
                html.P(item['description'], style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '0.9rem'}),
                html.Small(f"Published: {item['pub_date']}", style={'color': '#95a5a6', 'fontSize': '0.8rem'})
            ], style={
                'padding': '15px',
                'marginBottom': '10px',
                'backgroundColor': 'white',
                'borderRadius': '5px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            })
            news_cards.append(news_card)
        
        news_section = html.Div(news_cards)
        
        return fig, metrics, performance_metrics, volume_fig, news_section
        
    except Exception as e:
        logger.error(f"Error in update_dashboard: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(text="Error displaying data", showarrow=False, font=dict(size=16))
        error_metrics = html.Div("Error loading metrics")
        
        return error_fig, error_metrics, "", go.Figure(), ""

# ======================
# RUN APPLICATION
# ======================
if __name__ == "__main__":
    # Environment configuration
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    
    # Run the application
    app.run_server(
        debug=Config.DEBUG, 
        host="0.0.0.0", 
        port=Config.PORT,
        dev_tools_ui=Config.DEBUG,
        dev_tools_props_check=Config.DEBUG
    )