import numpy as np
import matplotlib.pyplot as plt  # ⬅️ IMPORT CORRIGIDO
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class AnalisadorFotoCalibrado:
    def __init__(self):
        self.espectro_referencia = self._criar_espectro_referencia()
        self.fatores_calibracao = {}

    def _criar_espectro_referencia(self):
        """Cria um espectro de referência para calibração"""
        wavelengths = np.linspace(380, 750, 1000)  # nm
        espectro = []

        for wl in wavelengths:
            if wl < 440:
                r = -(wl - 440) / (440 - 380);
                g = 0.0;
                b = 1.0
            elif wl < 490:
                r = 0.0;
                g = (wl - 440) / (490 - 440);
                b = 1.0
            elif wl < 510:
                r = 0.0;
                g = 1.0;
                b = -(wl - 510) / (510 - 490)
            elif wl < 580:
                r = (wl - 510) / (580 - 510);
                g = 1.0;
                b = 0.0
            elif wl < 645:
                r = 1.0;
                g = -(wl - 645) / (645 - 580);
                b = 0.0
            else:
                r = 1.0;
                g = 0.0;
                b = 0.0

            rgb = np.array([r, g, b])
            espectro.append(np.clip(rgb, 0, 1))

        return {
            'wavelengths': wavelengths,
            'colors': np.array(espectro)
        }

    def calibrar_com_simulacao(self, resultados_simulacao, caminho_foto, orientacao='horizontal'):
        """
        Calibra a foto usando os resultados da simulação como referência
        """
        print("Iniciando calibração...")

        # Carregar foto
        self.carregar_foto(caminho_foto, orientacao)

        # Obter dados da simulação
        posicoes_sim = resultados_simulacao['x_cm']

        # Analisar foto sem calibração
        resultados_foto = self.analisar_comprimentos_onda(comprimento_total_cm=5.0, suavizar=False)
        comprimentos_foto = resultados_foto['comprimentos_onda_nm']

        # Usar método simplificado para encontrar escala
        melhor_escala = self._encontrar_melhor_escala(comprimentos_foto)

        print(f"Comprimento médio foto: {np.mean(comprimentos_foto):.1f} nm")
        print(f"Variação foto: {np.max(comprimentos_foto) - np.min(comprimentos_foto):.1f} nm")

        # Aplicar calibração
        self.fatores_calibracao = {
            'escala': melhor_escala,
            'deslocamento': 0,
            'comprimento_calibrado_cm': 5.0
        }

        print(f"Calibração aplicada: Escala = {melhor_escala:.4f}")
        return self.fatores_calibracao

    def _encontrar_melhor_escala(self, comprimentos_foto):
        """Encontra a melhor escala de forma simplificada"""
        # Calcular variação dos comprimentos de onda
        variacao = np.max(comprimentos_foto) - np.min(comprimentos_foto)

        print(f"Variação de comprimentos de onda na foto: {variacao:.1f} nm")

        # Escalas baseadas na variação observada
        if variacao > 300:  # Muita variação
            return 0.7
        elif variacao > 200:
            return 0.9
        elif variacao < 50:  # Pouca variação
            return 1.5
        elif variacao < 100:
            return 1.3
        else:
            return 1.1  # Caso médio

    def carregar_foto(self, caminho_arquivo, orientacao='horizontal'):
        """
        Carrega a foto usando PIL (sem OpenCV)
        """
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

    def analisar_comprimentos_onda_calibrado(self, suavizar=True):
        """
        Analisa os comprimentos de onda com calibração aplicada
        """
        if not self.fatores_calibracao:
            raise ValueError("Execute calibrar_com_simulacao() primeiro")

        # Analisar sem calibração
        resultados_brutos = self.analisar_comprimentos_onda(
            comprimento_total_cm=5.0,
            suavizar=False
        )

        # Aplicar calibração
        escala = self.fatores_calibracao['escala']

        # Ajustar posições
        posicoes_calibradas = resultados_brutos['posicoes_cm'] * escala
        posicoes_calibradas = np.clip(posicoes_calibradas, 0, 5)

        # Reamostrar para ter pontos uniformes em 0-5 cm
        posicoes_finais = np.linspace(0, 5, len(posicoes_calibradas))
        interp_func = interp1d(posicoes_calibradas, resultados_brutos['comprimentos_onda_nm'],
                               kind='linear', fill_value='extrapolate')
        comprimentos_calibrados = interp_func(posicoes_finais)

        # Suavizar se solicitado
        if suavizar and len(comprimentos_calibrados) > 11:
            comprimentos_calibrados = savgol_filter(comprimentos_calibrados, 11, 3)

        self.resultados_calibrados = {
            'posicoes_cm': posicoes_finais,
            'comprimentos_onda_nm': comprimentos_calibrados,
            'confiancas': resultados_brutos['confiancas'],
            'cores_originais': resultados_brutos['cores_originais'],
            'fatores_calibracao': self.fatores_calibracao
        }

        print(f"✅ Análise calibrada concluída: {len(comprimentos_calibrados)} pontos")
        return self.resultados_calibrados

    def analisar_comprimentos_onda(self, comprimento_total_cm=5.0, suavizar=True):
        """
        Analisa os comprimentos de onda em cada posição
        """
        posicoes_cm = np.linspace(0, comprimento_total_cm, self.num_pixels)
        comprimentos_onda = []
        confiancas = []

        print(f"Analisando {self.num_pixels} pixels...")

        for i, pixel in enumerate(self.linha_pixels):
            lambda_nm, confianca = self._encontrar_melhor_comprimento_onda(pixel)
            comprimentos_onda.append(lambda_nm)
            confiancas.append(confianca)

            # Progresso
            if i % 100 == 0:
                print(f"Progresso: {i}/{self.num_pixels} pixels")

        comprimentos_onda = np.array(comprimentos_onda)
        confiancas = np.array(confiancas)

        print(f"Comprimentos de onda encontrados: {np.min(comprimentos_onda):.1f} - {np.max(comprimentos_onda):.1f} nm")

        if suavizar:
            mascara_valida = confiancas > 0.1
            if np.sum(mascara_valida) > 10:
                x_valido = posicoes_cm[mascara_valida]
                y_valido = comprimentos_onda[mascara_valida]

                if len(x_valido) > 3:
                    interp_func = interp1d(x_valido, y_valido, kind='linear',
                                           fill_value='extrapolate')
                    comprimentos_onda = interp_func(posicoes_cm)

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

    def plotar_comparacao(self, resultados_simulacao, titulo="Comparação: Simulação vs Foto Calibrada"):
        """Plota comparação entre simulação e foto calibrada"""
        if not hasattr(self, 'resultados_calibrados'):
            raise ValueError("Execute analisar_comprimentos_onda_calibrado() primeiro")

        # Dados da foto calibrada
        pos_foto = self.resultados_calibrados['posicoes_cm']
        comp_foto = self.resultados_calibrados['comprimentos_onda_nm']

        # Gráfico simples
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(pos_foto, comp_foto, 'ro-', markersize=2, linewidth=2, label='Foto Calibrada')
        plt.xlabel('Posição (cm) - Escala 0-5')
        plt.ylabel('Comprimento de Onda (nm)')
        plt.title('Análise da Foto\nComprimentos de Onda vs Posição')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.imshow([self.resultados_calibrados['cores_originais']], aspect='auto', extent=[0, 5, 0, 1])
        plt.xlabel('Posição (cm)')
        plt.ylabel('Cores Originais')
        plt.yticks([])
        plt.title('Padrão de Cores da Foto')

        plt.tight_layout()
        plt.show()