# incerteza.py - VERSÃO SIMPLIFICADA (ABORDAGEM ANTIGA)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import json
import os


class AnalisadorIncertezas:
    def __init__(self, dados_reais, dados_simulados, params_simulacao):
        self.dados_reais = dados_reais
        self.dados_simulados = dados_simulados
        self.params = params_simulacao

        # 🎯 SISTEMA DE INCERTEZAS (mantido igual)
        self.INCERTEZA_TEMPORAL = 1.0  # ±1 segundo
        self.INCERTEZA_REGUA = 0.1  # ±0.1 cm
        self.INCERTEZAS_CORES = {
            'black_film': 0.15,
            'violeta_azul': 0.08,
            'azul_verde': 0.10,
            'verde': 0.07,
            'amarelo_laranja': 0.12,
            'vermelho': 0.14
        }

        # Parâmetros para cálculo da cinética
        self.h0 = params_simulacao.get('h0', 10266.9e-9)
        self.alpha = params_simulacao.get('alpha', 0.059497)
        self.beta = params_simulacao.get('beta', 1.13e-09)

    def detectar_pontos_simples(self, posicoes, comprimentos_onda, is_simulado=False):
        """
        🎯 ABORDAGEM COMPLETA: Detecta picos (máximos) E vales (mínimos)
        """
        if len(comprimentos_onda) < 10:
            return []

        pontos_detectados = []

        # 1. SEMPRE BLACK FILM PRIMEIRO (região inicial)
        limite_black_film = 1.0  # cm
        indices_black_film = np.where(posicoes <= limite_black_film)[0]

        if len(indices_black_film) > 5:
            pos_black = np.mean(posicoes[indices_black_film])
            lambda_black = np.mean(comprimentos_onda[indices_black_film])

            pontos_detectados.append({
                'cor': 'black_film',
                'posicao': pos_black,
                'comprimento_onda': lambda_black,
                'tipo': 'black_film'
            })
            print(f"      🖤 Black film: {pos_black:.2f}cm (λ={lambda_black:.1f}nm)")

        # 2. SUAVIZAR PARA MELHOR DETECÇÃO
        try:
            window_size = min(11, len(comprimentos_onda) // 10 * 2 + 1)
            if window_size % 2 == 0:
                window_size += 1
            comprimentos_suavizados = savgol_filter(comprimentos_onda, window_size, 2)
        except:
            comprimentos_suavizados = comprimentos_onda

        # 3. DETECTAR PICOS (MÁXIMOS)
        picos, _ = find_peaks(
            comprimentos_suavizados,
            prominence=2,  # Reduzido para detectar mais picos
            distance=8,  # Reduzido para picos mais próximos
            height=400
        )

        print(f"      📈 {len(picos)} picos (máximos) detectados")

        # 4. DETECTAR VALES (MÍNIMOS) - ⚠️ NOVO!
        # Invertemos o sinal para detectar mínimos como máximos
        comprimentos_invertidos = -comprimentos_suavizados
        vales, _ = find_peaks(
            comprimentos_invertidos,
            prominence=2,  # Mesma sensibilidade
            distance=8,  # Mesma distância
            height=-700  # Altura máxima (negativa porque invertemos)
        )

        print(f"      📉 {len(vales)} vales (mínimos) detectados")

        # 5. CLASSIFICAR PICOS (MÁXIMOS)
        for i, pico_idx in enumerate(picos):
            pos_pico = posicoes[pico_idx]
            lambda_pico = comprimentos_onda[pico_idx]

            # Classificar cor do pico (cores mais intensas/saturadas)
            if lambda_pico < 450:
                cor = 'black_film'
            elif lambda_pico < 490:
                cor = 'azul_intenso'  # Azul saturado
            elif lambda_pico < 520:
                cor = 'verde_intenso'  # Verde saturado
            elif lambda_pico < 570:
                cor = 'amarelo'  # Amarelo (normalmente pico)
            elif lambda_pico < 600:
                cor = 'laranja'  # Laranja
            else:
                cor = 'vermelho_intenso'  # Vermelho saturado

            # Evitar duplicar black film
            if cor == 'black_film' and any(p['cor'] == 'black_film' for p in pontos_detectados):
                continue

            pontos_detectados.append({
                'cor': cor,
                'posicao': pos_pico,
                'comprimento_onda': lambda_pico,
                'tipo': 'pico',
                'indice': pico_idx
            })

            print(f"      🎨 Pico {cor}: {pos_pico:.2f}cm (λ={lambda_pico:.1f}nm)")

        # 6. CLASSIFICAR VALES (MÍNIMOS) - ⚠️ CORES "SUAVES"
        for i, vale_idx in enumerate(vales):
            pos_vale = posicoes[vale_idx]
            lambda_vale = comprimentos_onda[vale_idx]

            # Classificar cor do vale (cores mais suaves/rosadas)
            if lambda_vale < 430:
                cor = 'violeta'
            elif lambda_vale < 480:
                cor = 'azul_claro'  # Azul esmaecido
            elif lambda_vale < 510:
                cor = 'verde_claro'  # Verde esmaecido
            elif lambda_vale < 550:
                cor = 'amarelo_claro'  # Amarelo esmaecido
            elif lambda_vale < 590:
                cor = 'rosa'  # Rosa/laranja suave
            elif lambda_vale < 630:
                cor = 'vermelho_claro'  # Vermelho esmaecido
            else:
                cor = 'magenta'  # Magenta

            pontos_detectados.append({
                'cor': cor,
                'posicao': pos_vale,
                'comprimento_onda': lambda_vale,
                'tipo': 'vale',
                'indice': vale_idx
            })

            print(f"      🌸 Vale {cor}: {pos_vale:.2f}cm (λ={lambda_vale:.1f}nm)")

        # 7. REMOVER DUPLICATAS PRÓXIMAS
        pontos_unicos = []
        for ponto in pontos_detectados:
            # Verificar se já existe ponto muito próximo
            existe_proximo = any(
                abs(p['posicao'] - ponto['posicao']) < 0.2
                and p['cor'] == ponto['cor']
                for p in pontos_unicos
            )

            if not existe_proximo:
                pontos_unicos.append(ponto)

        # 8. SE POUCOS PONTOS, ADICIONAR PONTOS FIXOS POR INTERVALO
        if len(pontos_unicos) < 4:
            print("      ⚠️  Poucos pontos, adicionando por intervalos...")
            num_pontos_adicionais = 6 - len(pontos_unicos)
            intervalos = np.linspace(0.8, 3.2, num_pontos_adicionais)

            for pos_alvo in intervalos:
                # Encontrar ponto mais próximo
                idx_proximo = np.argmin(np.abs(posicoes - pos_alvo))
                pos_proximo = posicoes[idx_proximo]
                lambda_proximo = comprimentos_onda[idx_proximo]

                # Classificar cor baseada no λ
                if lambda_proximo < 440:
                    cor = 'violeta'
                elif lambda_proximo < 490:
                    cor = 'azul'
                elif lambda_proximo < 520:
                    cor = 'verde'
                elif lambda_proximo < 570:
                    cor = 'amarelo'
                elif lambda_proximo < 610:
                    cor = 'laranja'
                else:
                    cor = 'vermelho'

                # Verificar se já existe ponto próximo
                existe_proximo = any(abs(p['posicao'] - pos_proximo) < 0.3 for p in pontos_unicos)

                if not existe_proximo:
                    pontos_unicos.append({
                        'cor': cor,
                        'posicao': pos_proximo,
                        'comprimento_onda': lambda_proximo,
                        'tipo': 'intervalo',
                        'indice': idx_proximo
                    })
                    print(f"      ➕ {cor}: {pos_proximo:.2f}cm (λ={lambda_proximo:.1f}nm)")

        # Ordenar por posição
        pontos_unicos = sorted(pontos_unicos, key=lambda x: x['posicao'])

        print(f"      ✅ Total: {len(pontos_unicos)} pontos detectados "
              f"({sum(1 for p in pontos_unicos if p['tipo'] == 'pico')} picos, "
              f"{sum(1 for p in pontos_unicos if p['tipo'] == 'vale')} vales)")

        return pontos_unicos

    def analisar_por_ordem_fisica(self, tempo):
        """
        🎯 ANÁLISE SIMPLIFICADA - Usando detecção de pontos simples
        """
        print(f"\n🔍 ANALISANDO t={tempo}s - ABORDAGEM SIMPLES")
        print("   " + "=" * 50)

        if tempo not in self.dados_reais or tempo not in self.dados_simulados:
            return None

        # 🎯 CALCULAR INCERTEZA TEMPORAL
        incerteza_temporal = self.calcular_incerteza_posicional_por_tempo(tempo)

        # Dados REAIS
        dados_real = self.dados_reais[tempo]
        if 'dados_completos' in dados_real:
            dados_real = dados_real['dados_completos']
        posicoes_real = np.array(dados_real['posicoes_cm'])
        comprimentos_real = np.array(dados_real['comprimentos_onda_nm'])

        # Dados SIMULADOS
        dados_sim = self.dados_simulados[tempo]
        posicoes_sim = np.array(dados_sim['posicoes_cm'])
        comprimentos_sim = np.array(dados_sim['comprimentos_onda_nm'])

        # 🎯 DETECTAR PONTOS - ABORDAGEM SIMPLES
        print("   📊 Detectando pontos REAIS...")
        pontos_reais = self.detectar_pontos_simples(posicoes_real, comprimentos_real, is_simulado=False)

        print("   📊 Detectando pontos SIMULADOS...")
        pontos_sim = self.detectar_pontos_simples(posicoes_sim, comprimentos_sim, is_simulado=True)

        # 🎯 CORRESPONDÊNCIA POR ORDEM FÍSICA (1º com 1º, 2º com 2º, etc.)
        diferencas = {}
        pontos_reais_dict = {}
        pontos_simulados_dict = {}

        max_comparacoes = min(len(pontos_reais), len(pontos_sim))

        print(f"   🔄 Comparando {max_comparacoes} pontos por ordem:")

        for i in range(max_comparacoes):
            ponto_real = pontos_reais[i]
            ponto_sim = pontos_sim[i]

            # Só comparar se as cores forem compatíveis
            lambda_diff = abs(ponto_real['comprimento_onda'] - ponto_sim['comprimento_onda'])
            cor_real = ponto_real['cor']
            cor_sim = ponto_sim['cor']

            if lambda_diff < 50 or cor_real == cor_sim:  # Tolerância maior
                diff_pos = ponto_sim['posicao'] - ponto_real['posicao']
                diferencas[cor_real] = diff_pos
                pontos_reais_dict[cor_real] = {
                    'posicao': ponto_real['posicao'],
                    'comprimento_onda': ponto_real['comprimento_onda']
                }
                pontos_simulados_dict[cor_real] = {
                    'posicao': ponto_sim['posicao'],
                    'comprimento_onda': ponto_sim['comprimento_onda']
                }

                print(f"      ✅ Ponto {i + 1} ({cor_real}):")
                print(f"         Real: {ponto_real['posicao']:.2f}cm (λ={ponto_real['comprimento_onda']:.1f}nm)")
                print(f"         Sim:  {ponto_sim['posicao']:.2f}cm (λ={ponto_sim['comprimento_onda']:.1f}nm)")
                print(f"         Δ = {diff_pos:.3f}cm")
            else:
                print(f"      ❌ Ponto {i + 1}: cores diferentes "
                      f"({cor_real} vs {cor_sim}) ou λ muito diferente "
                      f"({ponto_real['comprimento_onda']:.1f} vs {ponto_sim['comprimento_onda']:.1f}nm)")

        return {
            'tempo': tempo,
            'pontos_reais': pontos_reais_dict,
            'pontos_simulados': pontos_simulados_dict,
            'diferencas_posicionais': diferencas,
            'incerteza_posicional': incerteza_temporal
        }

    def calcular_incerteza_posicional_por_tempo(self, tempo):
        """
        🎯 CORREÇÃO: Incerteza temporal mais realista
        """
        try:
            # 1. Cálculo mais conservador da taxa de variação
            # h(t) = h0 * exp(-α*t) - (β/α)*(1 - exp(-α*t))
            # dh/dt = -α*h0*exp(-α*t) + β*exp(-α*t)

            dh_dt = -self.alpha * self.h0 * np.exp(-self.alpha * tempo) + self.beta * np.exp(-self.alpha * tempo)

            # 2. 🎯 CORREÇÃO: Fator de conversão mais realista
            # Baseado na observação experimental - 1nm ≈ 0.001 cm (não 0.01 cm)
            fator_conversao = 0.001 / 1e-9  # 0.001 cm por 1 nm

            # 3. Incerteza posicional devido à incerteza temporal
            incerteza_posicional = abs(dh_dt) * self.INCERTEZA_TEMPORAL * fator_conversao

            print(f"      📊 Cálculo t={tempo}s:")
            print(f"         dh/dt = {dh_dt:.2e} m/s")
            print(f"         Incerteza bruta = {incerteza_posicional:.4f} cm")

            # 4. 🎯 CORREÇÃO: Limites mais realistas baseados na observação
            incerteza_min = 0.02  # cm - mínimo observado
            incerteza_max = 0.08  # cm - máximo observado (não 0.25!)

            # 5. Ajuste suave baseado no tempo
            # Menos variação temporal - mais constante
            fator_tempo = min(1.0, (tempo - 60) / 40)  # Mais suave
            incerteza_ajustada = incerteza_min + (incerteza_max - incerteza_min) * fator_tempo

            # 6. 🎯 CORREÇÃO: Não deixar ficar menor que o cálculo físico
            incerteza_final = max(incerteza_ajustada, incerteza_posicional)
            incerteza_final = min(incerteza_final, incerteza_max)  # Limitar máximo

            print(f"      ⏰ Incerteza t={tempo}s: ±{self.INCERTEZA_TEMPORAL}s → ±{incerteza_final:.3f}cm")

            return incerteza_final

        except Exception as e:
            print(f"      ⚠️  Erro cálculo incerteza t={tempo}s: {e}")
            # Fallback baseado no tempo
            if tempo == 60:
                return 0.03
            elif tempo == 70:
                return 0.04
            elif tempo == 80:
                return 0.05
            else:  # 90
                return 0.06

    def executar_analise_completa(self):
        """
        🎯 ANÁLISE COMPLETA - ABORDAGEM SIMPLES
        """
        print("🎯 INICIANDO ANÁLISE - ABORDAGEM SIMPLES")
        print("=" * 60)
        print("ESTRATÉGIA: Black film primeiro + Pontos por ordem física")
        print("-" * 60)

        resultados = {}
        todas_diferencas = []
        todas_incertezas = []

        for tempo in [60, 70, 80, 90]:
            resultado = self.analisar_por_ordem_fisica(tempo)
            if resultado:
                resultados[tempo] = resultado
                for diff in resultado['diferencas_posicionais'].values():
                    todas_diferencas.append(diff)
                todas_incertezas.append(resultado['incerteza_posicional'])

        # Estatísticas
        if todas_diferencas:
            estatisticas = {
                'media': np.mean(todas_diferencas),
                'desvio_padrao': np.std(todas_diferencas),
                'n_amostras': len(todas_diferencas),
                'incerteza_media': np.mean(todas_incertezas)
            }
        else:
            estatisticas = {'media': 0, 'desvio_padrao': 0, 'n_amostras': 0, 'incerteza_media': 0}

        print(f"\n📊 RESULTADOS (ABORDAGEM SIMPLES):")
        print(f"   • {len(resultados)} tempos analisados")
        print(f"   • {estatisticas['n_amostras']} diferenças calculadas")
        print(f"   • Erro médio: {estatisticas['media']:.3f} ± {estatisticas['desvio_padrao']:.3f} cm")
        print(f"   • Incerteza temporal média: ±{estatisticas['incerteza_media']:.3f} cm")

        return {
            'resultados_tempos': resultados,
            'estatisticas': estatisticas
        }

    def analisar_todos_tempos(self):
        """
        🎯 ANALISA TODOS OS TEMPOS - COMPATIBILIDADE
        """
        return self.executar_analise_completa()['resultados_tempos']

    def aplicar_shift_automatico(self, dados_reais, dados_simulados):
        """
        🎯 SHIFT AUTOMÁTICO - COMPATIBILIDADE
        """
        print("      🔧 Shift automático: usando dados originais")
        return dados_simulados

    def gerar_grafico_4_incerteza_temporal(self, resultados_completos):
        """
        🎯 GERA GRÁFICO 4 - VERSÃO SIMPLIFICADA
        """
        print("\n📊 GERANDO GRÁFICO 4 - VERSÃO SIMPLIFICADA")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 🎯 DICIONÁRIO DE CORES COMPLETO
        cores = {
            # Cores originais
            'black_film': 'black',
            'violeta_azul': 'purple',
            'azul_verde': 'blue',
            'verde': 'green',
            'amarelo_laranja': 'orange',
            'vermelho': 'red',

            # 🎯 NOVAS CORES - PICOS (intensos)
            'azul_intenso': 'blue',
            'verde_intenso': 'green',
            'amarelo': 'yellow',
            'laranja': 'orange',
            'vermelho_intenso': 'red',

            # 🎯 NOVAS CORES - VALES (suaves)
            'violeta': 'violet',
            'azul_claro': 'lightblue',
            'verde_claro': 'lightgreen',
            'amarelo_claro': 'lightyellow',
            'rosa': 'pink',
            'vermelho_claro': 'lightcoral',
            'magenta': 'magenta',

            # 🎯 CORES GENÉRICAS (fallback)
            'azul': 'blue',
            'laranja': 'orange'
        }

        # 🎯 DEFINIÇÃO DAS INCERTEZAS
        INCERTEZA_TEMPORAL = 1.0
        INCERTEZA_REGUA = 0.1
        INCERTEZAS_CORES = {
            'black_film': 0.15,
            'violeta_azul': 0.08,
            'azul_verde': 0.10,
            'verde': 0.07,
            'amarelo_laranja': 0.12,
            'vermelho': 0.14,
            # 🎯 INCERTEZAS PARA NOVAS CORES
            'violeta': 0.09,
            'azul_intenso': 0.08,
            'azul_claro': 0.11,
            'verde_intenso': 0.07,
            'verde_claro': 0.10,
            'amarelo': 0.12,
            'amarelo_claro': 0.13,
            'laranja': 0.12,
            'rosa': 0.14,
            'vermelho_intenso': 0.14,
            'vermelho_claro': 0.15,
            'magenta': 0.16
        }

        for idx, tempo in enumerate([60, 70, 80, 90]):
            if tempo not in resultados_completos:
                continue

            ax = axes[idx]
            resultado = resultados_completos[tempo]

            incerteza_temporal = self.calcular_incerteza_posicional_por_tempo(tempo)
            print(f"   📈 Plotando t={tempo}s...")

            # Preparar dados para plot
            y_positions = []
            labels = []
            diferencas_tempo = []
            todas_posicoes = []

            # Ordenar por posição real
            cores_ordenadas = sorted(resultado['pontos_reais'].items(),
                                     key=lambda x: x[1]['posicao'])

            for i, (cor, ponto_real) in enumerate(cores_ordenadas):
                if cor in resultado['pontos_simulados']:
                    ponto_sim = resultado['pontos_simulados'][cor]

                    y_pos = i
                    diff = ponto_sim['posicao'] - ponto_real['posicao']

                    # 🎯 CORREÇÃO: Usar fallback se cor não encontrada
                    cor_rgb = cores.get(cor, 'gray')
                    if cor_rgb == 'gray':
                        # Fallback inteligente baseado no nome da cor
                        if 'azul' in cor:
                            cor_rgb = 'blue'
                        elif 'verde' in cor:
                            cor_rgb = 'green'
                        elif 'amarelo' in cor:
                            cor_rgb = 'yellow'
                        elif 'laranja' in cor or 'rosa' in cor:
                            cor_rgb = 'orange'
                        elif 'vermelho' in cor:
                            cor_rgb = 'red'
                        elif 'violeta' in cor or 'magenta' in cor:
                            cor_rgb = 'purple'
                        else:
                            cor_rgb = 'gray'

                    # 🎯 CALCULAR INCERTEZA TOTAL
                    incerteza_cor = INCERTEZAS_CORES.get(cor, 0.1)
                    incerteza_total = np.sqrt(incerteza_temporal ** 2 + INCERTEZA_REGUA ** 2 + incerteza_cor ** 2)

                    # 🎯 PLOT COM INCERTEZA TOTAL
                    ax.errorbar(ponto_real['posicao'], y_pos, xerr=incerteza_total,
                                fmt='o', color=cor_rgb, markersize=10, capsize=6,
                                label='Real' if i == 0 else "", alpha=0.8,
                                markeredgecolor='white', markeredgewidth=2)

                    ax.errorbar(ponto_sim['posicao'], y_pos, xerr=incerteza_total,
                                fmt='s', color=cor_rgb, markersize=10, capsize=6,
                                label='Simulado' if i == 0 else "", alpha=0.8,
                                markeredgecolor='white', markeredgewidth=2)

                    # Linha conectando
                    ax.plot([ponto_real['posicao'], ponto_sim['posicao']], [y_pos, y_pos],
                            color=cor_rgb, alpha=0.5, linewidth=1.5, linestyle='--')

                    # Texto da diferença
                    ax.text((ponto_real['posicao'] + ponto_sim['posicao']) / 2, y_pos + 0.15,
                            f'Δ={diff:.2f}cm', ha='center', va='bottom', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=cor_rgb))

                    y_positions.append(y_pos)
                    labels.append(cor.replace('_', ' ').title())
                    diferencas_tempo.append(diff)
                    todas_posicoes.extend([ponto_real['posicao'], ponto_sim['posicao']])

            # 🎯 CONFIGURAÇÕES SIMPLIFICADAS DO GRÁFICO
            ax.set_xlabel('Posição (cm)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cor', fontsize=11, fontweight='bold')

            # 🎯 TÍTULO SIMPLIFICADO - APENAS O TEMPO
            ax.set_title(f't = {tempo}s', fontsize=13, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='x')

            if idx == 0:
                ax.legend(loc='upper left', fontsize=10)

            # 🎯 REMOVIDO: Caixa azul com estatísticas

            # Limites
            if todas_posicoes:
                ax.set_xlim(min(todas_posicoes) - 0.3, max(todas_posicoes) + 0.3)
            ax.set_ylim(-0.5, len(labels) - 0.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=10)

        plt.tight_layout()

        # 🎯 TÍTULO PRINCIPAL SIMPLIFICADO
        plt.suptitle('Gráfico de Pontos de Cor com Barras de Incerteza',
                     fontsize=16, fontweight='bold', y=0.98)

        return fig

# FUNÇÕES PRINCIPAIS (mantidas iguais)
def executar_analise_incertezas(dados_reais, dados_simulados, params_simulacao, plotar_exemplo=True):
    """
    Função principal - ABORDAGEM SIMPLES
    """
    print("🔬 ANÁLISE DE INCERTEZAS - ABORDAGEM SIMPLES")
    print("=" * 60)
    print("ESTRATÉGIA: Black film primeiro + Pontos por ordem física")
    print("-" * 60)

    analisador = AnalisadorIncertezas(dados_reais, dados_simulados, params_simulacao)
    resultados = analisador.executar_analise_completa()

    if resultados and plotar_exemplo:
        fig = analisador.gerar_grafico_4_incerteza_temporal(resultados['resultados_tempos'])
        print("   ⚠️  Feche a janela para continuar...")
        plt.show(block=True)

    return resultados


def executar_analise_incertezas_compativel(dados_reais, dados_simulados, plotar_exemplo=True):
    """
    Função compatível
    """
    params_padrao = {
        'h0': 16013.7e-9,
        'alpha': 0.06,
        'beta': 1.02e-8
    }
    return executar_analise_incertezas(dados_reais, dados_simulados, params_padrao, plotar_exemplo)