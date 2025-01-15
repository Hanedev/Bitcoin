import streamlit as st


from app.data_handler import fetch_bitcoin_data, prepare_data_for_lstm
from app.calculator import train_lstm_model, predict_future_values
from app.plotter import plot_predictions
import pandas as pd
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les données
st.title("Simulateur et prédictions de gains Bitcoin")
try:
    bitcoin_data = fetch_bitcoin_data()
    if bitcoin_data.empty:
        st.error("Les données du Bitcoin n'ont pas pu être récupérées.")
        st.stop()
except Exception as e:
    logging.error(f"Erreur lors de la récupération des données: {e}")
    st.error("Une erreur s'est produite lors de la récupération des données.")
    st.stop()

# Sidebar pour les prédictions
st.sidebar.header("Paramètres des prédictions")
future_days = st.sidebar.slider("Nombre de jours à prédire", 30, 365, 90, step=30)
invested_amount = st.sidebar.number_input("Somme à investir ($)", min_value=10.0, step=10.0)

# Paramètres avancés
st.sidebar.header("Paramètres avancés")
look_back = st.sidebar.slider("Longueur de la séquence (look back)", 30, 120, 60, step=10)
epochs = st.sidebar.slider("Nombre d'époques", 5, 50, 10, step=5)
batch_size = st.sidebar.slider("Taille des batches", 16, 128, 32, step=16)

if st.sidebar.button("Lancer les prédictions"):
    try:
        # Préparer les données
        X, y, scaler = prepare_data_for_lstm(bitcoin_data, look_back)
        model = train_lstm_model(X, y, epochs=epochs, batch_size=batch_size)

        # Faire les prédictions
        predictions = predict_future_values(model, scaler, bitcoin_data, future_days, look_back)
        future_dates = pd.date_range(bitcoin_data["timestamp"].max(), periods=future_days + 1, freq='D')[1:]
        future_data = pd.DataFrame({"Date": future_dates, "Prix prédit ($)": predictions})

        # Calcul de la valeur future de l'investissement
        current_price = bitcoin_data["close"].iloc[-1]
        future_values = (invested_amount / current_price) * predictions
        future_data["Valeur de l'investissement ($)"] = future_values

        # Affichage
        st.write(future_data)
        plt = plot_predictions(bitcoin_data, future_dates, predictions)
        st.pyplot(plt)

    except Exception as e:
        logging.error(f"Erreur lors des prédictions: {e}")
        st.error("Une erreur s'est produite lors des prédictions. Veuillez réessayer.")
