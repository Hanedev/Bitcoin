from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

def train_lstm_model(X, y, epochs=10, batch_size=32):
    """
    Entraîne un modèle LSTM.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict_future_values(model, scaler, data, future_days, look_back=60):
    """
    Prédit les valeurs futures en utilisant le modèle LSTM.
    """
    scaled_data = scaler.transform(data[['close']].values)
    last_sequence = scaled_data[-look_back:]
    predictions = []

    for _ in range(future_days):
        pred = model.predict(last_sequence.reshape(1, look_back, 1))
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
