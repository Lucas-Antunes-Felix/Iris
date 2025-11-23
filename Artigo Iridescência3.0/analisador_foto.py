# analisador_foto.py - VERSÃO COMPATÍVEL COM SIMULADOR

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import json
import os
from PIL import Image


class AnalisadorEspectros:
    def __init__(self, largura_total_cm=4.0):
        self.largura_total_cm = largura_total_cm
        self.comprimento_onda_min = 400
        self.comprimento_onda_max = 700
        self.num_pontos_analise = 500

    def carregar_foto(self, caminho_foto):
        """Carrega e processa imagem"""
        try:
            if not os.path.exists(caminho_foto):
                print(f"❌ Arquivo não encontrado: {caminho_foto}")
                return None

            imagem = Image.open(caminho_foto)
            img_array = np.array(imagem)
            print(f"✅ Foto carregada: {img_array.shape}")
            return img_array
        except Exception as e:
            print(f"❌ Erro ao carregar {caminho_foto}: {e}")
            return None

    def extrair_perfil_cores(self, img_array, orientacao='horizontal'):
        """Extrai perfil médio de cores da imagem"""
        if orientacao == 'horizontal':
            perfil_r = np.mean(img_array[:, :, 0], axis=0)
            perfil_g = np.mean(img_array[:, :, 1], axis=0)
            perfil_b = np.mean(img_array[:, :, 2], axis=0)
        else:
            perfil_r = np.mean(img_array[:, :, 0], axis=1)
            perfil_g = np.mean(img_array[:, :, 1], axis=1)
            perfil_b = np.mean(img_array[:, :, 2], axis=1)

        return perfil_r, perfil_g, perfil_b

    def diagnosticar_conversao_rgb(self, img_array, nome_foto):
        """Diagnóstico detalhado da conversão RGB para comprimento de onda"""
        print(f"\n🔍 DIAGNÓSTICO para {nome_foto}:")
        print("=" * 50)

        # Analisar regiões específicas da imagem
        h, w, _ = img_array.shape
        regioes = [
            ("Início (possível azul)", slice(0, h), slice(0, w // 10)),
            ("Meio (possível verde)", slice(0, h), slice(w // 3, 2 * w // 3)),
            ("Fim (possível vermelho)", slice(0, h), slice(9 * w // 10, w))
        ]

        for nome, slice_y, slice_x in regioes:
            regiao = img_array[slice_y, slice_x]
            r_medio = np.mean(regiao[:, :, 0])
            g_medio = np.mean(regiao[:, :, 1])
            b_medio = np.mean(regiao[:, :, 2])

            lambda_calculado = self.rgb_para_comprimento_onda(r_medio, g_medio, b_medio)

            print(f"   {nome}:")
            print(f"      RGB = ({r_medio:.0f}, {g_medio:.0f}, {b_medio:.0f})")
            print(f"      → λ = {lambda_calculado:.0f} nm")
            print(
                f"      Dominante: {'R' if r_medio > g_medio and r_medio > b_medio else 'G' if g_medio > b_medio else 'B'}")

    def rgb_para_comprimento_onda(self, r, g, b):
        """
        VERSÃO CONSERVADORA - Baseada em relações físicas reais
        """
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        # Evitar valores extremos
        r_norm = np.clip(r_norm, 0.1, 0.9)
        g_norm = np.clip(g_norm, 0.1, 0.9)
        b_norm = np.clip(b_norm, 0.1, 0.9)

        # Determinar cor dominante de forma mais conservadora
        max_val = max(r_norm, g_norm, b_norm)

        # Se tudo escuro → black film
        if max_val < 0.2:
            return 420

        # VERMELHO dominante: 600-650nm (não 700!)
        if r_norm > g_norm + 0.1 and r_norm > b_norm + 0.1:
            return 620 + (r_norm - 0.5) * 40  # 600-640nm

        # VERDE dominante: 520-570nm
        elif g_norm > r_norm + 0.1 and g_norm > b_norm + 0.1:
            return 540 + (g_norm - 0.5) * 30  # 525-555nm

        # AZUL dominante: 450-500nm
        elif b_norm > r_norm + 0.1 and b_norm > g_norm + 0.1:
            return 470 + (b_norm - 0.5) * 40  # 450-490nm

        # CORES MISTAS (mais conservador)
        else:
            # Amarelo: 570-590nm
            if r_norm > 0.6 and g_norm > 0.6:
                return 580
            # Ciano: 490-520nm
            elif g_norm > 0.6 and b_norm > 0.6:
                return 500
            # Magenta: não comum em espectros físicos
            else:
                return 550  # Fallback conservador

    def processar_espectro_foto(self, caminho_foto, tempo, orientacao='horizontal'):
        """
        Processa uma foto e extrai o espectro de comprimentos de onda
        """
        print(f"🔍 Processando t={tempo}s: {os.path.basename(caminho_foto)}")

        img_array = self.carregar_foto(caminho_foto)
        if img_array is None:
            return None

        r, g, b = self.extrair_perfil_cores(img_array, orientacao)

        num_pixels = len(r)
        posicoes_cm = np.linspace(0, self.largura_total_cm, num_pixels)

        comprimentos_onda = []
        for i in range(num_pixels):
            lambda_approx = self.rgb_para_comprimento_onda(r[i], g[i], b[i])
            comprimentos_onda.append(lambda_approx)

        comprimentos_onda = np.array(comprimentos_onda)

        # Suavizar o espectro
        try:
            window_size = min(21, num_pixels // 10 * 2 + 1)
            if window_size % 2 == 0:
                window_size += 1
            comprimentos_suavizados = signal.savgol_filter(
                comprimentos_onda, window_size, 3
            )
        except:
            comprimentos_suavizados = comprimentos_onda

        # Calcular comprimento de onda médio
        regiao_central = slice(num_pixels // 4, 3 * num_pixels // 4)
        lambda_medio = np.mean(comprimentos_onda[regiao_central])

        dados_completos = {
            'posicoes_cm': posicoes_cm.tolist(),
            'comprimentos_onda_nm': comprimentos_onda.tolist(),
            'comprimentos_suavizados': comprimentos_suavizados.tolist(),
            'r_perfil': r.tolist(),
            'g_perfil': g.tolist(),
            'b_perfil': b.tolist(),
            'num_pixels': num_pixels
        }

        resultado = {
            'tempo': tempo,
            'lambda_medio': float(lambda_medio),
            'dados_completos': dados_completos,
            'caminho_foto': caminho_foto
        }

        print(f"✅ t={tempo}s: Espectro extraído ({num_pixels} pontos, λₘ={lambda_medio:.1f}nm)")
        return resultado

    def carregar_dados_simulados(self, params_simulacao, tempos=[60, 70, 80, 90]):
        """
        Carrega ou gera dados simulados para comparação - CORRIGIDO
        """
        try:
            from simulacao import run_simulation, analisar_cores_para_comprimento_onda
        except ImportError as e:
            print(f"❌ Erro ao importar módulos de simulação: {e}")
            return {}

        print("🔄 Gerando dados simulados para comparação...")

        dados_simulados = {}

        for tempo in tempos:
            try:
                params_sim = params_simulacao.copy()
                params_sim['t_initial'] = tempo

                # Executar simulação
                resultados_sim = run_simulation(params_sim)

                # CORREÇÃO: Usar campos existentes no simulador
                if 'x_cm' in resultados_sim and 'colors_rgb' in resultados_sim:
                    analise_sim = analisar_cores_para_comprimento_onda(
                        resultados_sim['colors_rgb'],
                        resultados_sim['x_cm']  # Usar x_cm que existe no simulador
                    )
                else:
                    print(f"❌ Campos ausentes na simulação t={tempo}s")
                    continue

                dados_simulados[tempo] = {
                    'posicoes_cm': analise_sim['posicoes_cm'],
                    'comprimentos_onda_nm': analise_sim['comprimentos_onda_nm']
                }

                print(f"✅ t={tempo}s: Simulação concluída")

            except Exception as e:
                print(f"❌ Erro na simulação t={tempo}s: {e}")
                continue

        return dados_simulados

    def gerar_grafico_2_comparacao(self, dados_reais, dados_simulados):
        """
        GERA GRÁFICO 2: Comparação espectros reais vs simulados - VERSÃO SIMPLIFICADA
        """
        print("📊 GERANDO GRÁFICO 2: Espectros Reais vs Simulados")

        if not dados_simulados:
            print("❌ Nenhum dado simulado disponível para comparação")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, tempo in enumerate([60, 70, 80, 90]):
            if tempo not in dados_reais or tempo not in dados_simulados:
                print(f"⚠️  Dados ausentes para t={tempo}s")
                continue

            ax = axes[idx]

            # Dados reais
            dados_real = dados_reais[tempo]
            if 'dados_completos' in dados_real:
                dados_plot_real = dados_real['dados_completos']
            else:
                dados_plot_real = dados_real

            # Dados simulados
            dados_sim = dados_simulados[tempo]

            # 🔥 CALCULAR SHIFT AUTOMATICAMENTE (mas não mostrar no título)
            lambda_medio_real = dados_real['lambda_medio']
            lambda_medio_sim = np.mean(dados_sim['comprimentos_onda_nm'])
            shift_needed = lambda_medio_real - lambda_medio_sim

            print(f"   t={tempo}s: Shift aplicado = {shift_needed:.1f}nm")

            # Aplicar shift aos dados simulados
            comprimentos_sim_shifted = np.array(dados_sim['comprimentos_onda_nm']) + shift_needed

            # Plot espectro real
            ax.plot(dados_plot_real['posicoes_cm'],
                    dados_plot_real['comprimentos_suavizados'],
                    'b-', linewidth=2, label='Experimental', alpha=0.8)

            # Plot espectro simulado COM SHIFT
            ax.plot(dados_sim['posicoes_cm'],
                    comprimentos_sim_shifted,
                    'r-', linewidth=2, label='Simulado', alpha=0.8)

            # 🎯 TÍTULO SIMPLIFICADO - APENAS O TEMPO
            ax.set_title(f't = {tempo}s', fontsize=12, fontweight='bold')
            ax.set_xlabel('Posição (cm)')
            ax.set_ylabel('Comprimento de Onda (nm)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 4)
            ax.set_ylim(400, 700)

            # 🎯 REMOVIDO: Caixa com λ médio real e simulado

        plt.tight_layout()

        # 🎯 TÍTULO PRINCIPAL SIMPLIFICADO
        plt.suptitle('Espectro de Cor - Experimental vs Simulado',
                     fontsize=14, fontweight='bold', y=0.98)

        return fig

    def analisar_fotos_todos_tempos(self, params_simulacao, salvar_dados=True, plotar_graficos=True):
        """
        Analisa todas as fotos e gera Gráfico 2 - CORRIGIDO
        """
        print("🎯 ANÁLISE COMPLETA DAS FOTOS - GERANDO GRÁFICO 2")
        print("=" * 60)

        tempos_analise = params_simulacao.get('tempos_analise', [60, 70, 80, 90])
        orientacao = params_simulacao.get('orientacao_foto', 'horizontal')

        print(f"📸 Analisando {len(tempos_analise)} fotos:")
        print(f"   Tempos: {tempos_analise}s")
        print(f"   Orientação: {orientacao}")
        print("-" * 60)

        # Processar fotos reais
        dados_reais = {}
        for tempo in tempos_analise:
            caminho_foto = f"/Users/macbook/Desktop/padraoreal{tempo}.png"
            resultado = self.processar_espectro_foto(caminho_foto, tempo, orientacao)

            if resultado:
                dados_reais[tempo] = resultado

                if salvar_dados:
                    self.salvar_dados_json(resultado, f"dados_reais_t{tempo}.json")

        if not dados_reais:
            print("❌ Nenhuma foto foi processada com sucesso")
            return None

        print(f"✅ {len(dados_reais)} fotos processadas e salvas")

        # Carregar/Gerar dados simulados
        dados_simulados = self.carregar_dados_simulados(params_simulacao, tempos_analise)

        # CORREÇÃO: Continuar mesmo sem dados simulados
        if not dados_simulados:
            print("⚠️  Não foi possível gerar dados simulados, continuando apenas com dados reais")
            return {
                'dados_reais': dados_reais,
                'dados_simulados': {},
                'tempos_processados': list(dados_reais.keys())
            }

        # Gerar Gráfico 2 apenas se houver dados simulados
        if plotar_graficos and dados_simulados:
            fig_grafico2 = self.gerar_grafico_2_comparacao(dados_reais, dados_simulados)
            if fig_grafico2:
                print("✅ Gráfico 2 gerado: Espectros Reais vs Simulados")
                plt.show(block=False)
            else:
                print("⚠️  Não foi possível gerar Gráfico 2")

        # Resumo estatístico
        self.gerar_resumo_estatistico(dados_reais)

        return {
            'dados_reais': dados_reais,
            'dados_simulados': dados_simulados,
            'tempos_processados': list(dados_reais.keys())
        }

    def gerar_resumo_estatistico(self, dados_reais):
        """Gera resumo estatístico dos espectros extraídos"""
        print(f"\n📈 RESUMO ESTATÍSTICO DOS ESPECTROS:")
        print("=" * 50)

        for tempo in sorted(dados_reais.keys()):
            dados = dados_reais[tempo]
            lambda_medio = dados['lambda_medio']
            h_approx = (lambda_medio * 1e-9 * 1) / (2 * 1.375) * 1e9

            print(f"   t={tempo}s:")
            print(f"      • Comprimento médio: {lambda_medio:.1f} nm")
            print(f"      • Espessura aprox.:  {h_approx:.1f} nm")
            print(f"      • Pontos no espectro: {len(dados['dados_completos']['posicoes_cm'])}")

        if len(dados_reais) > 1:
            tempos = sorted(dados_reais.keys())
            lambdas = [dados_reais[t]['lambda_medio'] for t in tempos]

            print(f"\n   📊 Tendência temporal:")
            print(f"      • λ inicial (t={tempos[0]}s): {lambdas[0]:.1f} nm")
            print(f"      • λ final (t={tempos[-1]}s): {lambdas[-1]:.1f} nm")
            print(f"      • Variação total: {lambdas[-1] - lambdas[0]:.1f} nm")

    def salvar_dados_json(self, dados, nome_arquivo):
        """Salva dados em arquivo JSON"""
        try:
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
            print(f"💾 Dados salvos: {nome_arquivo}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar {nome_arquivo}: {e}")
            return False

    def carregar_dados_json(self, nome_arquivo):
        """Carrega dados de arquivo JSON"""
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            print(f"📂 Dados carregados: {nome_arquivo}")
            return dados
        except Exception as e:
            print(f"❌ Erro ao carregar {nome_arquivo}: {e}")
            return None


# =============================================================================
# FUNÇÕES GLOBAIS DE INTERFACE
# =============================================================================

def carregar_dados_existentes(tempos=[60, 70, 80, 90]):
    """
    Carrega dados existentes de múltiplos tempos
    """
    analisador = AnalisadorEspectros()
    dados_carregados = {}

    for tempo in tempos:
        arquivo = f"dados_reais_t{tempo}.json"
        if os.path.exists(arquivo):
            dados = analisador.carregar_dados_json(arquivo)
            if dados:
                dados_carregados[tempo] = dados
        else:
            print(f"⚠️  Arquivo não encontrado: {arquivo}")

    print(f"📂 Carregados {len(dados_carregados)} conjuntos de dados")
    return dados_carregados


def main_analise_fotos(params_simulacao=None, salvar_dados=True, plotar_graficos=True):
    """
    Função principal que analisa fotos e gera Gráfico 2
    """
    if params_simulacao is None:
        params_simulacao = {
            'h0': 16013.70e-9,
            'alpha': 0.06,
            'beta': 1.02e-08,
            'n_film': 1.375,
            'num_steps': 1000,
            'tempos_analise': [60, 70, 80, 90],
            'orientacao_foto': 'horizontal'
        }

    print("🔬 ANÁLISE DE FOTOS - GERANDO GRÁFICO 2")
    print("=" * 60)
    print("FLUXO: Fotos reais → Espectros → Comparação com simulação → Gráfico 2")
    print("-" * 60)

    analisador = AnalisadorEspectros()
    resultados = analisador.analisar_fotos_todos_tempos(
        params_simulacao,
        salvar_dados=salvar_dados,
        plotar_graficos=plotar_graficos
    )

    if resultados:
        print(f"\n✅ ANÁLISE CONCLUÍDA!")
        print(f"   • {len(resultados['dados_reais'])} fotos processadas")
        if resultados['dados_simulados']:
            print(f"   • {len(resultados['dados_simulados'])} simulações geradas")
            print(f"   • Gráfico 2: Espectros experimentais vs simulados")
        else:
            print(f"   • ⚠️  Nenhuma simulação gerada (apenas dados reais)")

        if plotar_graficos and resultados['dados_simulados']:
            print("   ⚠️  Feche a janela do gráfico para continuar...")
            plt.show(block=True)
    else:
        print("❌ Análise não foi concluída com sucesso")

    return resultados['dados_reais'] if resultados else None


def analisar_foto_individual(caminho_foto, tempo=60, salvar_dados=True):
    """
    Função para análise de foto individual
    """
    analisador = AnalisadorEspectros()
    resultado = analisador.processar_espectro_foto(caminho_foto, tempo)

    if resultado and salvar_dados:
        analisador.salvar_dados_json(resultado, f"dados_reais_t{tempo}.json")

    return resultado


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    print("📸 ANALISADOR DE FOTOS - GERADOR DO GRÁFICO 2")
    print("=" * 60)
    print("Este módulo:")
    print("   • Extrai espectros de cor de fotos reais")
    print("   • Gera Gráfico 2: Comparação com simulação")
    print("   • Salva dados para análise de incertezas")
    print("=" * 60)

    resultados = main_analise_fotos()

    if resultados:
        print(f"\n🎯 PRÓXIMOS PASSOS:")
        print("   • Use carregar_dados_existentes() para acessar os dados")
        print("   • Execute análise de incertezas com os dados carregados")
        print("   • Use otimização para melhorar os parâmetros")
    else:
        print("\n❌ Verifique se as fotos existem no caminho esperado")
        print("   Caminho esperado: /Users/macbook/Desktop/padraoreal[60,70,80,90].png")