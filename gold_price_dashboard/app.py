from flask import Flask, render_template, request
import pandas as pd
from model.gold_model import predict_gold, predict_next_7_days
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Gold Price.csv")
GRAPH_PATH = os.path.join(BASE_DIR, "static", "graph.png")

print("CSV file path:", CSV_PATH)

# ================= LOAD CSV =================
raw_data = pd.read_csv(CSV_PATH)
raw_data.columns = [c.lower().strip() for c in raw_data.columns]
print("CSV Columns:", raw_data.columns.tolist())

# ================= DETECT DATE COLUMN =================
date_col = None
for col in ['date', 'time', 'datetime', 'timestamp']:
    if col in raw_data.columns:
        date_col = col
        break

if date_col is None:
    raise Exception(" No date/time column found in CSV")

# ================= DETECT PRICE COLUMN =================
price_col = None
for col in ['close', 'price', 'closing_price']:
    if col in raw_data.columns:
        price_col = col
        break

if price_col is None:
    raise Exception("❌ No price column found in CSV")

# ================= PREPARE DATA =================
data = raw_data[[date_col, price_col]]
data.columns = ['Date', 'Price']

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna()

# ================= DAILY AGGREGATION =================
data['Date'] = data['Date'].dt.date
data = data.groupby('Date')['Price'].mean().reset_index()
data['Date'] = pd.to_datetime(data['Date'])

# ================= SORT =================
data = data.sort_values('Date').reset_index(drop=True)

print("Final Data Range:", data['Date'].min(), "→", data['Date'].max())

# ================= GRAPH FUNCTION =================
def generate_graph(plot_data, period, future_date, future_price):
    plt.figure(figsize=(10, 4))

    if period == "daily":
        plot_data = plot_data.tail(30)
        title = "Gold Price Trend (Last 30 Days)"
    elif period == "weekly":
        plot_data = plot_data.tail(180)
        title = "Gold Price Trend (Last 6 Months)"
    else:
        plot_data = plot_data.tail(365)
        title = "Gold Price Trend (Last 1 Year)"

    plt.plot(
        plot_data['Date'],
        plot_data['Price'],
        color='gold',
        linewidth=2,
        label="Historical Price"
    )

    plt.scatter(
        future_date,
        future_price,
        color='red',
        s=90,
        label="Predicted Price"
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

# ================= FLASK ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    forecast = None

    if request.method == "POST":
        #  USER INPUT
        year = int(request.form["year"])
        period = request.form["period"].lower().strip()

        #  YEAR FILTER
        filtered_data = data[data['Date'].dt.year == year]
        if filtered_data.empty:
            filtered_data = data  # safety fallback

        #  PERIOD → DAYS
        if period == "daily":
            days = 1
        elif period == "weekly":
            days = 7
        elif period == "monthly":
            days = 30
        else:
            days = 7   # default safe option

        #  PREDICTION (BACKEND ACCURACY USED, NOT SHOWN)
        predicted_price, future_date, _ = predict_gold(filtered_data, days)
        result = round(predicted_price, 2)

        #  GRAPH
        generate_graph(filtered_data, period, future_date, predicted_price)

        #  NEXT 7 DAYS
        forecast = predict_next_7_days(filtered_data)

    return render_template(
        "index.html",
        prediction=result,
        forecast=forecast
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
