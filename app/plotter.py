import matplotlib.pyplot as plt

def plot_predictions(data, future_dates, predictions):
    """
    Affiche les prédictions futures avec les données historiques.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamp"], data["close"], label="Historique")
    plt.plot(future_dates, predictions, label="Prédictions", linestyle="--", color="orange")
    plt.title("Prédictions du prix futur du Bitcoin")
    plt.xlabel("Date")
    plt.ylabel("Prix ($)")
    plt.legend()
    return plt
