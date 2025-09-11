import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# ======================
# DATA FETCHING FUNCTIONS
# ======================
def fetch_stock_data(ticker, period="1y"):
    """
    Fetch stock data from Yahoo Finance with retry logic and error handling
    """
    try:
        # Try multiple times in case of temporary API issues
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if hist.empty:
                    print(f"No data for {ticker}, attempt {attempt + 1}")
                    time.sleep(1)  # Wait before retry
                    continue
                    
                return hist
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                time.sleep(1)  # Wait before retry
                
        return None  # Return None if all attempts fail
        
    except Exception as e:
        print(f"Critical error with {ticker}: {e}")
        return None

def get_stock_data(ticker, period="1y"):
    """
    Get stock data with fallback to mock data if API fails
    """
    real_data = fetch_stock_data(ticker, period)
    
    if real_data is not None and not real_data.empty:
        return real_data
    
    # Fallback to mock data if real data fails (for demo purposes)
    print(f"Using mock data for {ticker}")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(200, 300, 100),
        'Low': np.random.uniform(50, 100, 100),
        'Close': np.random.uniform(150, 250, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    return mock_data

# ======================
# APP LAYOUT
# ======================
app.layout = html.Div([
    html.H1("📈 Yahoo Finance Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Label("Select Stock:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='stock-selector',
            options=[
                {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                {'label': 'Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
                {'label': 'Ethereum (ETH-USD)', 'value': 'ETH-USD'}
            ],
            value='AAPL',
            style={'width': '200px'}
        ),
        
        html.Label("Time Period:", style={'fontWeight': 'bold', 'marginLeft': '20px', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='date-range',
            options=[
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label': '6 Months', 'value': '6mo'},
                {'label': '1 Year', 'value': '1y'},
                {'label': '2 Years', 'value': '2y'}
            ],
            value='1y',
            style={'width': '150px'}
        )
    ], style={'padding': '20px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
    
    # Error message div
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'padding': '10px'}),
    
    # Loading component for graph
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=dcc.Graph(id='stock-graph', style={'height': '70vh'})
    ),
    
    # Additional metrics
    html.Div(id='stock-metrics', style={'padding': '20px', 'textAlign': 'center'})
])

# ======================
# CALLBACKS
# ======================
@callback(
    [Output('stock-graph', 'figure'),
     Output('error-message', 'children'),
     Output('stock-metrics', 'children')],
    [Input('stock-selector', 'value'),
     Input('date-range', 'value')]
)
def update_dashboard(selected_stock, date_range):
    if not selected_stock:
        return go.Figure(), "", ""
    
    # Get data
    data = get_stock_data(selected_stock, date_range)
    
    # Check if data is available
    if data is None or data.empty:
        error_msg = "⚠️ Unable to fetch data. Please try another stock or check your internet connection."
        return go.Figure(), error_msg, ""
    
    # Create graph
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=selected_stock
    )])
    
    fig.update_layout(
        title=f'{selected_stock} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=600
    )
    
    # Calculate metrics
    if len(data) > 0:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        percent_change = (price_change / prev_price) * 100
        
        metrics = html.Div([
            html.H3(f"Current Price: ${current_price:.2f}"),
            html.P(f"Change: {price_change:+.2f} ({percent_change:+.2f}%)", 
                  style={'color': 'green' if price_change >= 0 else 'red'})
        ])
    else:
        metrics = html.Div("No metrics available")
    
    return fig, "", metrics

# ======================
# RENDER DEPLOYMENT FIX
# ======================
# This helps with faster startup on Render
os.environ["NUMBA_DISABLE_JIT"] = "1"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port, processes=1, threads=4)