import matplotlib.pyplot as plt
import numpy as np

def grafico_incerteza(resultados):
    intensidade = resultados["intensidade"]
    tempo = resultados["tempo"]

    media = np.mean(intensidade)
    desvio = np.std(intensidade)

    plt.errorbar(tempo, intensidade, yerr=desvio, fmt="o", alpha=0.5)
    plt.axhline(media, color="r", linestyle="--", label=f"Média = {media:.2f}")
    plt.title("Gráfico de Incertezas")
    plt.xlabel("Tempo")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.show()
