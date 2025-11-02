# analisador_foto.py - VERSÃO CORRIGIDA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os

class AnalisadorFotoCalibrado:
    def __init__(self):
        self.espectro_referencia = self._criar_espectro_referencia()
        self.comprimento_total_cm = 4.0

    def _criar_espectro_referencia(self):
        """Cria um espectro de referência para calibração"""
        wavelengths = np.linspace(380, 750, 1000)
        espectro = []

        for wl in wavelengths:
            if wl < 440:
                r = -(wl - 440) / (440 - 380)
                g = 0.0
                b = 1.0
            elif wl < 490:
                r = 0.0
                g = (wl - 440) / (490 - 440)
                b = 1.0
            elif wl < 510:
                r = 0.0
                g = 1.0
                b = -(wl - 510) / (510 - 490)
            elif wl < 580:
                r = (wl - 510) / (580 - 510)
                g = 1.0
                b = 0.0
            elif wl < 645:
                r = 1.0
                g = -(wl - 645) / (645 - 580)
                b = 0.0
            else:
                r = 1.0
                g = 0.0
                b = 0.0

            rgb = np.array([r, g, b])
            espectro.append(np.clip(rgb, 0, 1))

        return {
            'wavelengths': wavelengths,
            'colors': np.array(espectro)
        }

    def carregar_foto(self, caminho_arquivo, orientacao='horizontal'):
        """Carrega a foto usando PIL"""
        self.imagem = Image.open(caminho_arquivo)
        self.array_imagem = np.array(self.imagem) / 255.0
        self.orientacao = orientacao

        print(f"Imagem carregada: {self.array_imagem.shape}")

        # Extrair linha de pixels para análise
        if orientacao == 'horizontal':
            altura = self.array_imagem.shape[0]
            self.linha_pixels = self.array_imagem[altura // 2, :, :3]
        else:
            largura = self.array_imagem.shape[1]
            self.linha_pixels = self.array_imagem[:, largura // 2, :3]

        self.num_pixels = len(self.linha_pixels)
        print(f"Pixels analisados: {self.num_pixels}")

    def analisar_comprimentos_onda(self, comprimento_total_cm=4.0, suavizar=True):
        """Analisa os comprimentos de onda em cada posição (0-4 cm)"""
        posicoes_cm = np.linspace(0, comprimento_total_cm, self.num_pixels)
        comprimentos_onda = []
        confiancas = []

        print(f"Analisando {self.num_pixels} pixels...")

        for i, pixel in enumerate(self.linha_pixels):
            lambda_nm, confianca = self._encontrar_melhor_comprimento_onda(pixel)
            comprimentos_onda.append(lambda_nm)
            confiancas.append(confianca)

        comprimentos_onda = np.array(comprimentos_onda)
        confiancas = np.array(confiancas)

        print(f"Comprimentos de onda: {np.min(comprimentos_onda):.1f} - {np.max(comprimentos_onda):.1f} nm")

        if suavizar and len(comprimentos_onda) > 11:
            comprimentos_onda = savgol_filter(comprimentos_onda, 11, 3)

        return {
            'posicoes_cm': posicoes_cm,
            'comprimentos_onda_nm': comprimentos_onda,
            'confiancas': confiancas,
            'cores_originais': self.linha_pixels
        }

    def _encontrar_melhor_comprimento_onda(self, cor_rgb):
        """Encontra o comprimento de onda que melhor corresponde à cor RGB"""
        melhor_lambda = 550
        menor_erro = float('inf')

        for i, cor_ref in enumerate(self.espectro_referencia['colors']):
            erro = np.sqrt(np.sum((cor_rgb - cor_ref) ** 2))

            if erro < menor_erro:
                menor_erro = erro
                melhor_lambda = self.espectro_referencia['wavelengths'][i]

        confianca = max(0, 1 - menor_erro / np.sqrt(3))
        return melhor_lambda, confianca

    def analisar_foto_individual(self, caminho_foto, tempo):
        """Analisa uma foto individual e retorna resultados"""
        print(f"\n{'=' * 50}")
        print(f"📊 ANALISANDO FOTO t={tempo}s")
        print(f"{'=' * 50}")

        try:
            # Carregar e analisar foto
            self.carregar_foto(caminho_foto)
            resultados = self.analisar_comprimentos_onda(comprimento_total_cm=4.0, suavizar=True)

            # Calcular estatísticas
            lambda_medio = np.mean(resultados['comprimentos_onda_nm'])
            lambda_min = np.min(resultados['comprimentos_onda_nm'])
            lambda_max = np.max(resultados['comprimentos_onda_nm'])

            print(f"✅ t={tempo}s → λ = {lambda_medio:.1f} nm (var: {lambda_min:.1f}-{lambda_max:.1f} nm)")

            return {
                'tempo': tempo,
                'caminho_foto': caminho_foto,
                'lambda_medio': lambda_medio,
                'lambda_min': lambda_min,
                'lambda_max': lambda_max,
                'dados_completos': resultados
            }

        except Exception as e:
            print(f"❌ Erro na análise da foto {caminho_foto}: {e}")
            return None

    def analisar_lote_simples(self, caminhos_fotos, tempos):
        """Analisa múltiplas fotos em lote"""
        resultados = {}

        for caminho, tempo in zip(caminhos_fotos, tempos):
            resultado = self.analisar_foto_individual(caminho, tempo)
            if resultado:
                resultados[tempo] = resultado

        return resultados

    def plotar_espectros_0a4(self, resultados_lote):
        """Plota os espectros das 4 fotos com X de 0 a 4 cm e Y com lambda"""
        plt.figure(figsize=(12, 8))

        cores = ['red', 'blue', 'green', 'orange']
        tempos = list(resultados_lote.keys())

        for i, (tempo, resultado) in enumerate(resultados_lote.items()):
            dados = resultado['dados_completos']
            plt.plot(dados['posicoes_cm'], dados['comprimentos_onda_nm'],
                     color=cores[i], linewidth=2, label=f't = {tempo}s')

        plt.xlabel('Posição (cm) - Escala 0-4 cm')
        plt.ylabel('Comprimento de Onda (nm)')
        plt.title('Espectros das 4 Fotografias Temporais\n(Posição vs Comprimento de Onda)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 4)
        plt.ylim(350, 650)
        plt.tight_layout()
        plt.show()

# ⬇️⬇️⬇️ FUNÇÃO PRINCIPAL - DEVE ESTAR FORA DA CLASSE ⬇️⬇️⬇️
def main_analise_fotos():
    """Função principal para analisar as 4 fotos temporais"""
    print("🎯 ANÁLISE DOS ESPECTROS TEMPORAIS")
    print("=" * 50)

    # Configurar caminhos das fotos
    area_trabalho = os.path.expanduser("~/Desktop")
    fotos = {
        60: os.path.join(area_trabalho, "padraoreal60.png"),
        70: os.path.join(area_trabalho, "padraoreal70.png"),
        80: os.path.join(area_trabalho, "padraoreal80.png"),
        90: os.path.join(area_trabalho, "padraoreal90.png")
    }

    # Verificar se fotos existem
    tempos = []
    caminhos_validos = []
    for tempo, caminho in fotos.items():
        if os.path.exists(caminho):
            tempos.append(tempo)
            caminhos_validos.append(caminho)
            print(f"✅ Foto t={tempo}s encontrada: {caminho}")
        else:
            print(f"❌ Foto não encontrada: {caminho}")

    if not caminhos_validos:
        print("❌ Nenhuma foto encontrada! Verifique os nomes dos arquivos.")
        return

    # Analisar fotos
    analisador = AnalisadorFotoCalibrado()
    resultados = analisador.analisar_lote_simples(caminhos_validos, tempos)

    # Mostrar resumo
    print(f"\n📋 RESUMO DOS RESULTADOS:")
    print("=" * 40)
    for tempo, resultado in resultados.items():
        print(f"t = {tempo}s: λ = {resultado['lambda_medio']:.1f} nm "
              f"({resultado['lambda_min']:.1f}-{resultado['lambda_max']:.1f} nm)")

    # Plotar gráfico
    analisador.plotar_espectros_0a4(resultados)

    return resultados

# Executar diretamente se o arquivo for rodado
if __name__ == "__main__":
    main_analise_fotos()