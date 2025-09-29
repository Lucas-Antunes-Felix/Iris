from simulacao import run_simulation
from analisador_foto import AnalisadorFotoCalibrado
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Parâmetros da simulação (os mesmos que geraram o padrão teórico)
params = {
    'h0': 1000e-9,
    'alpha': 1.1e-2,
    'beta': 0.9e-8,
    'n_film': 1.3,
    'num_steps': 1000,
    't_initial': 85
}


def main():
    # Executar simulação para obter referência
    print("Gerando simulação de referência...")
    resultados_simulacao = run_simulation(params)

    # Inicializar analisador calibrado
    analisador = AnalisadorFotoCalibrado()

    # Caminho da sua foto
    caminho_foto = "foto_padrao_real.jpg"  # Substitua pelo seu arquivo

    try:
        # Calibrar usando a simulação como referência
        print("Calibrando foto com simulação...")
        fatores_calibracao = analisador.calibrar_com_simulacao(
            resultados_simulacao,
            caminho_foto,
            orientacao='horizontal'  # Ajuste conforme sua foto
        )

        # Analisar com calibração aplicada
        print("Analisando foto calibrada...")
        resultados_calibrados = analisador.analisar_comprimentos_onda_calibrado(suavizar=True)

        # Plotar comparação
        analisador.plotar_comparacao(resultados_simulacao)

        # Gráfico final: Comprimentos de onda vs posição (0-5 cm calibrado)
        plt.figure(figsize=(10, 6))
        plt.plot(resultados_calibrados['posicoes_cm'],
                 resultados_calibrados['comprimentos_onda_nm'],
                 'r-', linewidth=3, label='Foto Calibrada')
        plt.xlabel('Posição (cm) - Escala Calibrada 0-5 cm')
        plt.ylabel('Comprimento de Onda (nm)')
        plt.title('Comprimentos de Onda vs Posição (Foto Calibrada)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Exportar dados calibrados
        dados_calibrados = np.column_stack([
            resultados_calibrados['posicoes_cm'],
            resultados_calibrados['comprimentos_onda_nm']
        ])
        np.savetxt('dados_foto_calibrada.csv', dados_calibrados,
                   delimiter=',', header='Posicao_cm,Comprimento_Onda_nm', fmt='%.6f')

        print("Análise concluída! Dados salvos em 'dados_foto_calibrada.csv'")

    except Exception as e:
        print(f"Erro: {e}")
        print("Verifique o caminho da foto e os parâmetros")


if __name__ == "__main__":
    main()