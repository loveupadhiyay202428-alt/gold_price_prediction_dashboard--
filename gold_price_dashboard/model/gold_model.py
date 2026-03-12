import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ================= SINGLE FUTURE PREDICTION =================
def predict_gold(data, days):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Price'].values

    model = LinearRegression()
    model.fit(X, y)

    # 🔮 Future prediction
    future_index = np.array([[len(data) + days]])
    future_price = model.predict(future_index)[0]

    future_date = data['Date'].iloc[-1] + pd.Timedelta(days=days)

    # 🔍 Accuracy (R2 Score)
    y_pred = model.predict(X)
    accuracy = r2_score(y, y_pred) * 100   # percentage

    return future_price, future_date, round(accuracy, 2)


# ================= NEXT 7 DAYS PREDICTION =================
def predict_next_7_days(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Price'].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = data['Date'].iloc[-1]

    future_days = np.arange(len(data) + 1, len(data) + 8).reshape(-1, 1)
    future_prices = model.predict(future_days)

    forecast = []
    for i, price in enumerate(future_prices, start=1):
        forecast.append({
            "date": (last_date + pd.Timedelta(days=i)).strftime("%d-%m-%Y"),
            "price": round(price, 2)
        })

    return forecast
