from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

def train_lstm_model(X, y):
    """
    Entraîne un modèle LSTM.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model

def simulate_lstm_predictions(scaled_data, future_days, look_back=60):
    """
    Simule des prédictions LSTM en extrapolant les valeurs historiques.
    """
    last_sequence = scaled_data[-look_back:]  # Dernière séquence de taille `look_back`
    predictions = []

    for _ in range(future_days):
        # Simule une prédiction en utilisant une simple moyenne des dernières valeurs
        predicted_value = np.mean(last_sequence[-look_back:])  # Moyenne de la dernière séquence
        predictions.append(predicted_value)  # Ajoute la prédiction
        
        # Mettre à jour la séquence avec la nouvelle valeur prédite
        last_sequence = np.append(last_sequence[1:], predicted_value)
    
    # Retourner les valeurs dans l'échelle originale
    return np.array(predictions)
