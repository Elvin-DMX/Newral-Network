import yfinance as yf
import plotly.express as px
import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
ticker = ['ADBE']


# In[50]:


df = yf.download(ticker,start = '2010-01-01', end='2022-12-31').dropna()
# Define the app
app = dash.Dash(__name__)

# Define the moving average periods
periods = [50, 100, 200]

# Calculate the moving averages
for period in periods:
    df[f"SMA{period}"] = df["Adj Close"].rolling(window=period).mean()

# Define the buy/sell signal
for i in range(len(df)):
    if df["SMA50"][i] > df["SMA100"][i] > df["SMA200"][i]:
        df.loc[df.index[i], "Signal"] = "Buy"
    elif df["SMA50"][i] < df["SMA100"][i] < df["SMA200"][i]:
        df.loc[df.index[i], "Signal"] = "Sell"
    else:
        df.loc[df.index[i], "Signal"] = "Hold"

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1(f"{ticker} Stock Price Analysis"),
    dcc.DatePickerRange(
        id="date-picker-range",
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max(),
        start_date=df.index.min(),
        end_date=df.index.max()
    ),
    dcc.Graph(
        id="stock-chart",
        figure={
            "data": [
                {"x": df.index, "y": df["Adj Close"], "type": "line", "name": "Price"},
                {"x": df.index, "y": df["SMA50"], "type": "line", "name": "SMA50"},
                {"x": df.index, "y": df["SMA100"], "type": "line", "name": "SMA100"},
                {"x": df.index, "y": df["SMA200"], "type": "line", "name": "SMA200"}
            ],
            "layout": {
                "title": f"{ticker} Stock Price",
                "yaxis": {"title": "Price"},
                "legend": {"orientation": "h", "y": -0.25}
            }
        }
    ),
    dcc.Graph(
        id="signal-chart",
        figure={
            "data": [
                {"x": df.index, "y": df["Signal"], "type": "scatter", "mode": "markers"}
            ],
            "layout": {
                "title": f"{ticker} Buy/Sell/Hold Signal",
                "yaxis": {"title": "Signal"}
            }
        }
    )
])

# Define the callback function for the date picker
@app.callback(
Output("stock-chart", "figure"),
Output("signal-chart", "figure"),
Input("date-picker-range", "start_date"),
Input("date-picker-range", "end_date")
)
def update_charts(start_date, end_date):
    filtered_df = df.loc[start_date:end_date]

    stock_chart = {
    "data": [
        {"x": filtered_df.index, "y": filtered_df["Adj Close"], "type": "line", "name": "Price"},
        {"x": filtered_df.index, "y": filtered_df["SMA50"], "type": "line", "name": "SMA50"},
        {"x": filtered_df.index, "y": filtered_df["SMA100"], "type": "line", "name": "SMA100"},
        {"x": filtered_df.index, "y": filtered_df["SMA200"], "type": "line", "name": "SMA200"}
    ],
    "layout": {
        "title": f"{ticker} Stock Price",
        "yaxis": {"title": "Price"},
        "legend": {"orientation": "h", "y": -0.25}
    }
   }

    signal_chart = {
    "data": [
        {"x": filtered_df.index, "y": filtered_df["Signal"], "type": "scatter", "mode": "markers"}
    ],
    "layout": {
        "title": f"{ticker} Buy/Sell/Hold Signal",
        "yaxis": {"title": "Signal"}
     }
}

    return stock_chart, signal_chart
if __name__ == "__main__":
    app.run_server()
