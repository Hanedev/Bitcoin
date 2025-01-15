import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fetch_bitcoin_data():
    """
    Récupère les données historiques du Bitcoin depuis l'API Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    start_time = 1288834974000  # Timestamp du 1er Bitcoin (2010)
    end_time = int(pd.Timestamp.now().timestamp() * 1000)  # Timestamp actuel en ms
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",  # Intervalle journalier
            "startTime": start_time,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"Erreur lors de la récupération des données : {response.text}")
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][6] + 1  # Prochain start_time après le dernier close_time

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "close"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["close"] = df["close"].astype(float)
    return df

def prepare_data_for_lstm(data, look_back=60):
    """
    Prépare les données pour un modèle LSTM.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']].values)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler
