import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta




# app.py
def _security_fingerprint():
    """7a8f9d3b-this-function-acts-as-legal-watermark"""
    return 0xDEADBEEF  # Unique hex code

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Tickers including S&P 500 and Crypto
TICKERS = [
    {'label': 'S&P 500 (^GSPC)', 'value': '^GSPC'},
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
    {'label': 'Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
    {'label': 'Ethereum (ETH-USD)', 'value': 'ETH-USD'}
]

# Set default dates (last 6 months)
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

app.layout = html.Div([
    html.H1("📈 Ultimate Trading Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Controls
    html.Div([
        dcc.Dropdown(
            id='ticker-dropdown',
            options=TICKERS,
            value='BTC-USD',
            clearable=False,
            style={'width': '300px', 'margin': '10px'}
        ),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=datetime(2015, 1, 1),
            max_date_allowed=end_date,
            start_date=start_date,
            end_date=end_date,
            display_format='YYYY-MM-DD'
        ),
        dcc.Checklist(
            id='indicator-toggle',
            options=[
                {'label': ' Show Bollinger Bands', 'value': 'BB'},
                {'label': ' Show Ichimoku Cloud', 'value': 'IC'},
                {'label': ' Show MACD', 'value': 'MACD'}
            ],
            value=['BB'],
            inline=True,
            style={'margin': '10px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),
    
    # Charts
    dcc.Graph(id='price-chart', style={'height': '60vh'}),
    dcc.Graph(id='secondary-chart', style={'height': '40vh'})
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('secondary-chart', 'figure')],
    [Input('ticker-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('indicator-toggle', 'value')]
)
def update_charts(ticker, start_date, end_date, indicators):
    try:
        # Download data
        end_date_adj = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date_adj, progress=False)
        
        if df.empty:
            raise Exception("No data returned from Yahoo Finance")
        
        # Clean columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # ====== INDICATOR CALCULATIONS ======
        # 1. Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Upper_Band'] = df['SMA20'] + 2*df['Close'].rolling(20).std()
        df['Lower_Band'] = df['SMA20'] - 2*df['Close'].rolling(20).std()
        
        # 2. MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 3. RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Ichimoku Cloud
        df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min())/2
        df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min())/2
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun'])/2).shift(26)
        df['Senkou_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min())/2).shift(26)
        
        # 5. ATR
        df['TR'] = pd.DataFrame({
            'HL': df['High'] - df['Low'],
            'HC': abs(df['High'] - df['Close'].shift()),
            'LC': abs(df['Low'] - df['Close'].shift())
        }).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # ====== VISUALIZATION ======
        price_fig = go.Figure()
        
        # Candlestick
        price_fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C'
        ))
        
        # Indicators based on toggle
        if 'BB' in indicators:
            price_fig.add_trace(go.Scatter(
                x=df.index, y=df['Upper_Band'],
                line=dict(color='rgba(150, 150, 150, 0.5)'),
                name='Upper BB',
                hoverinfo='none'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index, y=df['Lower_Band'],
                line=dict(color='rgba(150, 150, 150, 0.5)'),
                name='Lower BB',
                fill='tonexty',
                hoverinfo='none'
            ))
        
        if 'IC' in indicators:
            price_fig.add_trace(go.Scatter(
                x=df.index, y=df['Senkou_A'],
                line=dict(color='rgba(46, 204, 113, 0.4)'),
                name='Ichimoku A'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index, y=df['Senkou_B'],
                line=dict(color='rgba(231, 76, 60, 0.4)'),
                name='Ichimoku B',
                fill='tonexty'
            ))
        
        price_fig.update_layout(
            title=f'{ticker} Price Action',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            plot_bgcolor='#f9f9f9',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Secondary Chart (RSI + MACD)
        secondary_fig = go.Figure()
        
        # RSI
        secondary_fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI (14)',
            line=dict(color='#9B59B6')
        ))
        secondary_fig.add_hline(
            y=70, line_dash="dash", line_color="red",
            annotation_text="Overbought", annotation_position="bottom right"
        )
        secondary_fig.add_hline(
            y=30, line_dash="dash", line_color="green",
            annotation_text="Oversold", annotation_position="top right"
        )
        
        # MACD
        if 'MACD' in indicators:
            secondary_fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                name='MACD',
                line=dict(color='#3498DB'),
                yaxis='y2'
            ))
            secondary_fig.add_trace(go.Scatter(
                x=df.index, y=df['Signal'],
                name='Signal',
                line=dict(color='#F39C12'),
                yaxis='y2'
            ))
            
            secondary_fig.update_layout(
                yaxis2=dict(
                    title='MACD',
                    overlaying='y',
                    side='right'
                )
            )
        
        secondary_fig.update_layout(
            title='Momentum Indicators',
            yaxis_range=[0, 100],
            plot_bgcolor='#f9f9f9',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return price_fig, secondary_fig
        
    except Exception as e:
        print(f"Error: {str(e)}")
        error_fig = go.Figure()
        error_fig.update_layout(
            title="⚠️ Error Loading Data",
            annotations=[dict(
                text=f"Try: 1. Different dates 2. Another ticker 3. Check internet<br>Error: {str(e)}",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )]
        )
        return error_fig, error_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)




    #!/usr/bin/env python3
# PROPRIETARY CODE - © 2025 [ANUSHA  SAHA]. All rights reserved.
# Unauthorized copying, distribution, or use is strictly prohibited.