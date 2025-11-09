# otimizador_dados.py - VERSÃO INTELIGENTE COM PENALIDADES FORTES
"""
Módulo de otimização - VERSÃO SUPER RESTRITIVA:
- NUNCA aceita pioras na média global
- Penalidades MÁXIMAS para qualquer degradação
- Foca apenas em melhorias consistentes e reais
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from simulacao import run_simulation
from graficos import analisar_cores_para_comprimento_onda
import matplotlib.pyplot as plt
from incerteza import AnalisadorIncertezas
import os
import json


class OtimizadorCompleto:
    def __init__(self, dados_reais, params_simulacao):
        """
        Otimizador SUPER RESTRITIVO - Zero tolerância a pioras

        Args:
            dados_reais: dict com dados experimentais {tempo: dados}
            params_simulacao: dict com parâmetros de simulação
        """
        self.dados_reais = dados_reais
        self.params = params_simulacao
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # 🚨 CALCULAR ERRO INICIAL UMA ÚNICA VEZ
        self.erro_inicial_global = self.calcular_erro_super_restritivo(
            self.params['h0'], self.params['alpha'], self.params['beta']
        )
        print(f"🎯 ERRO INICIAL DE REFERÊNCIA: {self.erro_inicial_global:.1f}")

    def simular_com_parametros(self, h0, alpha, beta, tempos=None):
        """
        Simula para todos os tempos com parâmetros específicos
        Retorna dados simulados no formato padrão
        """
        if tempos is None:
            tempos = self.params['tempos_analise']

        dados_simulados = {}

        for tempo in tempos:
            try:
                params_sim = {
                    'h0': h0, 'alpha': alpha, 'beta': beta,
                    'n_film': self.params['n_film'],
                    'num_steps': self.params['num_steps'],
                    't_initial': tempo
                }

                resultados_sim = run_simulation(params_sim)
                analise_sim = analisar_cores_para_comprimento_onda(
                    resultados_sim['colors_rgb'],
                    resultados_sim['x_cm']
                )

                dados_simulados[tempo] = {
                    'posicoes_cm': analise_sim['posicoes_cm'],
                    'comprimentos_onda_nm': analise_sim['comprimentos_onda_nm']
                }

            except Exception as e:
                print(f"❌ Erro na simulação t={tempo}s: {e}")
                continue

        return dados_simulados

    def calcular_erro_super_restritivo(self, h0, alpha, beta):
        """
        VERSÃO SUPER RESTRITIVA: PENALIDADES MÁXIMAS para qualquer piora
        """
        try:
            plt.close('all')

            # 1. Simular
            dados_simulados = self.simular_com_parametros(h0, alpha, beta)
            dados_reais_formatados = {}
            for tempo, dados in self.dados_reais.items():
                if 'dados_completos' in dados:
                    dados_reais_formatados[tempo] = {
                        'posicoes_cm': dados['dados_completos']['posicoes_cm'],
                        'comprimentos_onda_nm': dados['dados_completos']['comprimentos_onda_nm']
                    }
                else:
                    dados_reais_formatados[tempo] = {
                        'posicoes_cm': dados['posicoes_cm'],
                        'comprimentos_onda_nm': dados['comprimentos_onda_nm']
                    }

            # 2. Análise
            analisador = AnalisadorIncertezas(dados_reais_formatados, dados_simulados)
            resultados_completos = analisador.analisar_todos_tempos()

            if not resultados_completos:
                return 10000  # Penalidade MUITO alta

            # 3. CORES CRÍTICAS (devem SEMPRE estar presentes)
            cores_criticas = ['azul', 'verde', 'amarelo', 'laranja']

            # 4. Coletar dados e calcular PENALIDADES MÁXIMAS
            todas_diferencas = []
            penalidade_total = 0

            for tempo in [60, 70, 80, 90]:
                if tempo in resultados_completos:
                    resultado = resultados_completos[tempo]

                    for cor in cores_criticas:
                        if cor in resultado['diferencas_posicionais']:
                            diff = resultado['diferencas_posicionais'][cor]
                            if diff is not None:
                                erro_absoluto = abs(diff)
                                todas_diferencas.append(erro_absoluto)

                                # 🚨 PENALIDADE MÁXIMA por erros grandes
                                if erro_absoluto > 0.8:  # > 0.8cm é INACEITÁVEL
                                    excesso = erro_absoluto - 0.8
                                    penalidade_total += excesso * 2000
                                    print(f"      🚨 ERRO GRANDE: {cor} em t={tempo}s: {erro_absoluto:.3f}cm → +{excesso * 2000:.0f}")
                        else:
                            # 🚨 PENALIDADE MÁXIMA por cor faltante
                            penalidade_total += 1000
                            print(f"      🚨 COR FALTANTE: {cor} em t={tempo}s → +1000")

            if not todas_diferencas:
                return 10000

            # 5. Calcular MÉDIA GLOBAL
            media_global = np.mean(todas_diferencas)
            max_erro = np.max(todas_diferencas)

            # 6. 🚨 PENALIDADE MÁXIMA se piorou em relação ao inicial
            if media_global > self.erro_inicial_global:
                piora = media_global - self.erro_inicial_global
                penalidade_piora = piora * 5000  # ⬅️ PENALIDADE MÁXIMA ABSOLUTA
                penalidade_total += penalidade_piora
                print(f"      📉 PIORA GLOBAL: {piora:.3f}cm → +{penalidade_piora:.0f}")

            # 7. ERRO TOTAL = Média + Penalidades MÁXIMAS
            erro_total = media_global * 100 + penalidade_total

            # Log a cada iteração
            if self.iteracao_atual % 2 == 0:
                print(f"      🔒 Iteração {self.iteracao_atual}: Média={media_global:.3f}cm, Penal={penalidade_total:.0f}")

            return erro_total

        except Exception as e:
            print(f"      ❌ Erro: {e}")
            return 10000

    def calcular_penalidades_restricoes(self, h0, alpha, beta):
        """
        Calcula penalidades baseadas em restrições físicas
        """
        penalidade = 0

        # 1. Restrições de positividade
        if h0 <= 0 or alpha <= 0 or beta <= 0:
            penalidade += 1000
            return penalidade

        # 2. Restrição h0: máximo físico
        if h0 > self.params.get('h0_max_observado', 20000e-9):
            excesso = (h0 - self.params['h0_max_observado']) / self.params['h0_max_observado']
            penalidade += 1000 * excesso

        # 3. Restrição alpha: faixa razoável
        alpha_min = self.params.get('alpha_min_observado', 0.01)
        alpha_max = self.params.get('alpha_max_observado', 0.1)

        if alpha < alpha_min:
            deficit = (alpha_min - alpha) / alpha_min
            penalidade += 500 * deficit
        elif alpha > alpha_max:
            excesso = (alpha - alpha_max) / alpha_max
            penalidade += 500 * excesso

        # 4. Restrição beta: faixa razoável
        beta_ideal = self.params.get('beta_ideal_observado', 1e-8)
        tolerancia_beta = self.params.get('tolerancia_beta_observado', 0.5)

        if abs(beta - beta_ideal) > tolerancia_beta * beta_ideal:
            desvio = abs(beta - beta_ideal) / beta_ideal
            penalidade += 300 * desvio

        return penalidade

    def funcao_custo(self, params):
        h0, alpha, beta = params

        # 1. Calcular erro SUPER RESTRITIVO
        erro_restritivo = self.calcular_erro_super_restritivo(h0, alpha, beta)

        # 2. Calcular penalidades físicas
        penalidade = self.calcular_penalidades_restricoes(h0, alpha, beta)

        # 3. Erro total
        erro_total = erro_restritivo + penalidade

        # 4. 🚨 ATUALIZAR MELHOR RESULTADO
        if erro_total < self.melhor_erro:
            self.melhor_erro = erro_total
            self.melhor_params = params.copy()
            print(f"      🎉 NOVO MELHOR! Erro: {erro_total:.1f}")

        # 5. Log do progresso
        self.iteracao_atual += 1
        status = "✨" if erro_total < self.melhor_erro else "➡️"

        print(f"   [{self.iteracao_atual:2d}] {status} Erro: {erro_total:.1f} = "
              f"{erro_restritivo:.1f} (restr) + {penalidade:.1f} (pen)")
        print(f"      h0={h0 * 1e9:.1f}nm, α={alpha:.4f}, β={beta:.2e}")
        print("-" * 60)

        return erro_total

    def executar_otimizacao(self):
        """
        Executa processo de otimização - ZERO TOLERÂNCIA A PIORAS
        """
        print("🎯 INICIANDO OTIMIZAÇÃO SUPER RESTRITIVA")
        print("=" * 60)
        print("ESTRATÉGIA: Zero tolerância a pioras - Penalidades MÁXIMAS")
        print(f"ERRO INICIAL DE REFERÊNCIA: {self.erro_inicial_global:.1f}")
        print("-" * 60)
        print("PARÂMETROS INICIAIS:")
        print(f"   h0: {self.params['h0'] * 1e9:.1f} nm")
        print(f"   α:  {self.params['alpha']:.4f}")
        print(f"   β:  {self.params['beta']:.2e}")
        print("-" * 60)

        # Resetar contadores
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # Ponto inicial
        x0 = [self.params['h0'], self.params['alpha'], self.params['beta']]

        # Limites físicos
        bounds = [
            (1e-9, 50000e-9),  # h0: 1nm a 50μm
            (1e-3, 1.0),       # alpha: 0.001 a 1.0
            (1e-12, 1e-6)      # beta: 1e-12 a 1e-6
        ]

        # Executar otimização
        resultado = minimize(
            self.funcao_custo,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': 60,
                'disp': True,
                'xatol': 1e-8,
                'fatol': 1e-6,
                'adaptive': True
            }
        )

        # 🚨 CRÍTICO: USAR SEMPRE OS MELHORES PARÂMETROS
        if self.melhor_params is not None and self.melhor_erro < float('inf'):
            print("✅ USANDO MELHORES PARÂMETROS DAS ITERAÇÕES")
            h0_opt, alpha_opt, beta_opt = self.melhor_params
            erro_final = self.melhor_erro
            melhorou = True
        else:
            print("⚠️  Nenhuma melhoria encontrada - usando parâmetros originais")
            h0_opt, alpha_opt, beta_opt = self.params['h0'], self.params['alpha'], self.params['beta']
            erro_final = self.erro_inicial_global
            melhorou = False

        print("=" * 60)
        print("✅ OTIMIZAÇÃO CONCLUÍDA!")
        print(f"   Iterações totais: {self.iteracao_atual}")
        print(f"   h0: {self.params['h0'] * 1e9:.1f} → {h0_opt * 1e9:.1f} nm")
        print(f"   α:  {self.params['alpha']:.4f} → {alpha_opt:.4f}")
        print(f"   β:  {self.params['beta']:.2e} → {beta_opt:.2e}")
        print(f"   Melhor erro encontrado: {self.melhor_erro:.1f}")

        # Parâmetros finais
        params_otimizados = {
            'h0': max(h0_opt, 1e-9),
            'alpha': max(alpha_opt, 1e-3),
            'beta': max(beta_opt, 1e-12),
            'n_film': self.params['n_film']
        }

        # 🎯 EXIBIR PARÂMETROS FINAIS
        print("\n" + "=" * 50)
        print("🎯 PARÂMETROS OTIMIZADOS FINAIS:")
        print("=" * 50)
        print(f"   h0 = {params_otimizados['h0'] * 1e9:.1f} nm")
        print(f"   α  = {params_otimizados['alpha']:.6f}")
        print(f"   β  = {params_otimizados['beta']:.2e}")
        print("=" * 50)

        return {
            'params_otimizados': params_otimizados,
            'resultado_otimizacao': resultado,
            'erro_final': erro_final,
            'melhorou': melhorou,
            'melhor_erro_encontrado': self.melhor_erro
        }

    def gerar_grafico_final_unico(self, params_antigos, params_otimizados):
        """
        Gera 1 GRÁFICO FINAL com o MESMO FORMATO do Gráfico 4
        Mostra ANTES vs DEPOIS da otimização
        """
        print("\n🎨 GERANDO GRÁFICO FINAL: Formato IDÊNTICO ao Gráfico 4")

        # 1. Simular dados ANTES e DEPOIS
        print("🔄 Simulando dados ANTES da otimização...")
        dados_sim_antigos = self.simular_com_parametros(
            params_antigos['h0'], params_antigos['alpha'], params_antigos['beta']
        )

        print("🔄 Simulando dados DEPOIS da otimização...")
        dados_sim_otimizados = self.simular_com_parametros(
            params_otimizados['h0'], params_otimizados['alpha'], params_otimizados['beta']
        )

        # 2. Preparar dados reais
        dados_reais_formatados = {}
        for tempo, dados in self.dados_reais.items():
            if 'dados_completos' in dados:
                dados_reais_formatados[tempo] = {
                    'posicoes_cm': dados['dados_completos']['posicoes_cm'],
                    'comprimentos_onda_nm': dados['dados_completos']['comprimentos_onda_nm']
                }
            else:
                dados_reais_formatados[tempo] = {
                    'posicoes_cm': dados['posicoes_cm'],
                    'comprimentos_onda_nm': dados['comprimentos_onda_nm']
                }

        # 3. Analisar ambos os casos
        print("📊 Analisando dados ANTES da otimização...")
        analisador_antes = AnalisadorIncertezas(dados_reais_formatados, dados_sim_antigos)
        resultados_antes_completos = analisador_antes.analisar_todos_tempos()

        print("📊 Analisando dados DEPOIS da otimização...")
        analisador_depois = AnalisadorIncertezas(dados_reais_formatados, dados_sim_otimizados)
        resultados_depois_completos = analisador_depois.analisar_todos_tempos()

        # 4. Coletar dados MÉDIOS de todos os tempos (60, 70, 80, 90s)
        INCERTEZA_REGUA = 0.05  # cm

        # Estrutura para armazenar posições médias por COR
        posicoes_reais_medias = {}  # {cor: posicao_real_media}
        posicoes_antes_medias = {}  # {cor: posicao_sim_antes_media}
        posicoes_depois_medias = {}  # {cor: posicao_sim_depois_media}
        contagens_por_cor = {}  # {cor: quantidade}

        # Cores na ordem de aparição (como no Gráfico 4)
        ordem_cores = ['black_film', 'violeta', 'azul', 'verde', 'amarelo', 'laranja', 'vermelho']

        for tempo in [60, 70, 80, 90]:
            if tempo in resultados_antes_completos and tempo in resultados_depois_completos:
                resultado_antes = resultados_antes_completos[tempo]
                resultado_depois = resultados_depois_completos[tempo]

                # Processar cada cor detectada
                for cor in ordem_cores:
                    # Verificar se a cor existe em AMBOS os resultados
                    if (cor in resultado_antes['pontos_reais'] and
                            cor in resultado_antes['pontos_simulados'] and
                            cor in resultado_depois['pontos_simulados']):

                        # Posição REAL (é a mesma para antes e depois)
                        pos_real = resultado_antes['pontos_reais'][cor]['posicao']

                        # Posição SIMULADA ANTES
                        pos_antes = resultado_antes['pontos_simulados'][cor]['posicao']

                        # Posição SIMULADA DEPOIS
                        pos_depois = resultado_depois['pontos_simulados'][cor]['posicao']

                        # Acumular para calcular média depois
                        if cor not in posicoes_reais_medias:
                            posicoes_reais_medias[cor] = 0
                            posicoes_antes_medias[cor] = 0
                            posicoes_depois_medias[cor] = 0
                            contagens_por_cor[cor] = 0

                        posicoes_reais_medias[cor] += pos_real
                        posicoes_antes_medias[cor] += pos_antes
                        posicoes_depois_medias[cor] += pos_depois
                        contagens_por_cor[cor] += 1

        # 5. Calcular médias finais
        cores_finais = []
        pos_reais_finais = []
        pos_antes_finais = []
        pos_depois_finais = []

        for cor in ordem_cores:
            if cor in contagens_por_cor and contagens_por_cor[cor] > 0:
                n = contagens_por_cor[cor]
                cores_finais.append(cor)
                pos_reais_finais.append(posicoes_reais_medias[cor] / n)
                pos_antes_finais.append(posicoes_antes_medias[cor] / n)
                pos_depois_finais.append(posicoes_depois_medias[cor] / n)

                # Log das médias
                diff_antes = pos_antes_finais[-1] - pos_reais_finais[-1]
                diff_depois = pos_depois_finais[-1] - pos_reais_finais[-1]
                melhoria = abs(diff_antes) - abs(diff_depois)

                print(f"   🎨 {cor}: Real {pos_reais_finais[-1]:.2f}cm → "
                      f"Antes {pos_antes_finais[-1]:.2f}cm (Δ={diff_antes:.3f}cm) → "
                      f"Depois {pos_depois_finais[-1]:.2f}cm (Δ={diff_depois:.3f}cm) → "
                      f"Melhoria: {melhoria:+.3f}cm")

        # 6. Criar GRÁFICO NO FORMATO DO GRÁFICO 4
        fig, ax = plt.subplots(figsize=(12, 8))

        # Cores para as features (igual ao Gráfico 4)
        cores_plot = {
            'black_film': 'black',
            'violeta': 'purple',
            'azul': 'blue',
            'verde': 'green',
            'amarelo': 'orange',
            'laranja': 'darkorange',
            'vermelho': 'red'
        }

        # Posições no eixo Y
        y_pos = np.arange(len(cores_finais))

        # Plotar cada cor na ORDEM DE APARIÇÃO
        labels_plot = []
        todas_posicoes = []

        for i, cor in enumerate(cores_finais):
            pos_real = pos_reais_finais[i]
            pos_antes = pos_antes_finais[i]
            pos_depois = pos_depois_finais[i]

            todas_posicoes.extend([pos_real, pos_antes, pos_depois])

            # Cor para plotagem
            cor_plot = cores_plot.get(cor, 'gray')

            # Label para o eixo Y
            if cor == 'black_film':
                label = 'Black Film'
            else:
                label = f'{cor.capitalize()}'
            labels_plot.append(label)

            # Ponto REAL (círculo preto)
            ax.errorbar(pos_real, y_pos[i],
                        xerr=INCERTEZA_REGUA, fmt='o', color='black',
                        markersize=12, capsize=8, capthick=2,
                        label='Real' if i == 0 else "",
                        alpha=0.9, markeredgecolor='white', markeredgewidth=2)

            # Ponto ANTES (quadrado vermelho)
            ax.errorbar(pos_antes, y_pos[i] - 0.15,
                        xerr=INCERTEZA_REGUA, fmt='s', color='red',
                        markersize=10, capsize=6, capthick=2,
                        label='Antes' if i == 0 else "",
                        alpha=0.8, markeredgecolor='white', markeredgewidth=1)

            # Ponto DEPOIS (quadrado azul)
            ax.errorbar(pos_depois, y_pos[i] + 0.15,
                        xerr=INCERTEZA_REGUA, fmt='s', color='blue',
                        markersize=10, capsize=6, capthick=2,
                        label='Otimizado' if i == 0 else "",
                        alpha=0.9, markeredgecolor='white', markeredgewidth=1)

            # Linhas conectando
            ax.plot([pos_real, pos_antes], [y_pos[i], y_pos[i] - 0.15],
                    'r--', alpha=0.6, linewidth=1)
            ax.plot([pos_real, pos_depois], [y_pos[i], y_pos[i] + 0.15],
                    'b-', alpha=0.8, linewidth=1.5)

            # Calcular melhorias
            diff_antes = pos_antes - pos_real
            diff_depois = pos_depois - pos_real
            melhoria = abs(diff_antes) - abs(diff_depois)
            percentual = (melhoria / abs(diff_antes)) * 100 if abs(diff_antes) > 0 else 0

            # Anotação da melhoria
            ax.text(max(pos_real, pos_antes, pos_depois) + 0.1, y_pos[i],
                    f'Melhoria: {melhoria:+.2f}cm\n({percentual:+.1f}%)',
                    ha='left', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9))

        # Configurações do gráfico (igual ao Gráfico 4)
        ax.set_xlabel('Posição (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cor', fontsize=12, fontweight='bold')
        ax.set_title('GRÁFICO 4: Comparação Antes vs Depois da Otimização\n' +
                     'Posições Médias com Incerteza da Régua',
                     fontsize=14, fontweight='bold', pad=20)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_plot, fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')

        # Ajustar limites do eixo X
        if todas_posicoes:
            x_min = min(todas_posicoes) - 0.2
            x_max = max(todas_posicoes) + 0.5
            ax.set_xlim(x_min, x_max)

        # Ajustar limites do eixo Y
        ax.set_ylim(-0.5, len(cores_finais) - 0.5)

        # Estatísticas gerais
        if len(cores_finais) > 1:  # Excluir black film
            erros_antes = [abs(pos_antes_finais[i] - pos_reais_finais[i]) for i in range(len(cores_finais)) if
                           cores_finais[i] != 'black_film']
            erros_depois = [abs(pos_depois_finais[i] - pos_reais_finais[i]) for i in range(len(cores_finais)) if
                            cores_finais[i] != 'black_film']

            if erros_antes and erros_depois:
                media_antes = np.mean(erros_antes)
                media_depois = np.mean(erros_depois)
                melhoria_geral = media_antes - media_depois
                percentual_geral = (melhoria_geral / media_antes) * 100 if media_antes > 0 else 0

                ax.text(0.02, 0.98,
                        f'ESTATÍSTICAS GERAIS:\n'
                        f'Erro médio ANTES: {media_antes:.3f} cm\n'
                        f'Erro médio DEPOIS: {media_depois:.3f} cm\n'
                        f'Melhoria: {melhoria_geral:.3f} cm ({percentual_geral:+.1f}%)',
                        transform=ax.transAxes, fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.9))

        plt.tight_layout()

        print("✅ GRÁFICO FINAL GERADO!")
        print(f"   • Formato IDÊNTICO ao Gráfico 4")
        print(f"   • {len(cores_finais)} cores incluindo black film")
        print(f"   • Melhoria geral: {melhoria_geral:.3f} cm ({percentual_geral:+.1f}%)")

        return fig

    def salvar_resultados(self, resultado_otimizacao, params_antigos):
        """
        Salva resultados da otimização - VERSÃO CORRIGIDA PARA JSON
        """
        params_opt = resultado_otimizacao['params_otimizados']

        # Converter todos os valores para tipos Python nativos
        dados_salvar = {
            'params_antigos': {
                'h0': float(params_antigos['h0']),
                'alpha': float(params_antigos['alpha']),
                'beta': float(params_antigos['beta'])
            },
            'params_otimizados': {
                'h0': float(params_opt['h0']),
                'alpha': float(params_opt['alpha']),
                'beta': float(params_opt['beta']),
                'n_film': float(params_opt['n_film'])
            },
            'erro_final': float(resultado_otimizacao['erro_final']),
            'iteracoes_totais': int(self.iteracao_atual),
            'melhorou': bool(resultado_otimizacao.get('melhorou', True))
        }

        with open('resultados_otimizacao.json', 'w', encoding='utf-8') as f:
            json.dump(dados_salvar, f, indent=2, ensure_ascii=False)

        print("💾 Resultados salvos em 'resultados_otimizacao.json'")

# =============================================================================
# FUNÇÃO PRINCIPAL SIMPLIFICADA
# =============================================================================

def executar_otimizacao_inteligente(dados_temporais, params_simulacao):
    """
    Função principal para otimização completa

    Args:
        dados_temporais: dados experimentais
        params_simulacao: parâmetros atuais de simulação

    Returns:
        dict com resultados completos da otimização
    """
    print("🚀 EXECUTANDO OTIMIZAÇÃO COMPLETA")
    print("=" * 60)
    print("ESTRATÉGIA: Solução balanceada para todas as cores em todos os tempos")
    print("-" * 60)

    # Parâmetros antigos
    params_antigos = {
        'h0': params_simulacao['h0'],
        'alpha': params_simulacao['alpha'],
        'beta': params_simulacao['beta']
    }

    # Criar otimizador
    otimizador = OtimizadorCompleto(dados_temporais, params_simulacao)

    # Executar otimização
    resultado_otimizacao = otimizador.executar_otimizacao()

    # Gerar APENAS 1 GRÁFICO FINAL
    grafico_final = otimizador.gerar_grafico_final_unico(params_antigos, resultado_otimizacao['params_otimizados'])

    # Salvar resultados
    otimizador.salvar_resultados(resultado_otimizacao, params_antigos)

    # Mostrar gráfico final
    plt.show(block=True)

    # Adicionar gráfico ao resultado final
    resultado_otimizacao['grafico_final'] = grafico_final

    return resultado_otimizacao


# =============================================================================
# FUNÇÃO DE COMPATIBILIDADE
# =============================================================================

def analisar_erros_manual(params_manuais, params_simulacao):
    """
    Função para análise manual (compatibilidade)
    """
    print("🔍 ANÁLISE MANUAL DE PARÂMETROS")
    print("=" * 50)

    # Carregar dados reais
    from analisador_foto import carregar_dados_existentes
    dados_reais = carregar_dados_existentes()

    if not dados_reais:
        print("❌ Dados reais não encontrados. Execute análise de fotos primeiro.")
        return None

    # Criar otimizador temporário para simulação
    otimizador = OtimizadorCompleto(dados_reais, params_simulacao)

    # Simular com parâmetros manuais
    dados_simulados = otimizador.simular_com_parametros(
        params_manuais['h0'],
        params_manuais['alpha'],
        params_manuais['beta']
    )

    # Preparar dados reais
    dados_reais_formatados = {}
    for tempo, dados in dados_reais.items():
        if 'dados_completos' in dados:
            dados_reais_formatados[tempo] = {
                'posicoes_cm': dados['dados_completos']['posicoes_cm'],
                'comprimentos_onda_nm': dados['dados_completos']['comprimentos_onda_nm']
            }

    # Executar análise de incertezas
    from incerteza import executar_analise_incertezas
    resultados = executar_analise_incertezas(dados_reais_formatados, dados_simulados, plotar_exemplo=True)

    return resultados


if __name__ == "__main__":
    print("⚙️  MÓDULO DE OTIMIZAÇÃO - VERSÃO FINAL")
    print("=" * 60)
    print("Este módulo:")
    print("   • Otimização balanceada para solução global")
    print("   • Exibe claramente os parâmetros otimizados no final")
    print("   • Gera 1 gráfico final no formato do Gráfico 4")
    print("=" * 60)

    # Exemplo de uso
    from analisador_foto import carregar_dados_existentes

    dados = carregar_dados_existentes()
    if dados:
        print("✅ Dados carregados. Use executar_otimizacao_inteligente(dados, params_simulacao)")
    else:
        print("❌ Execute análise de fotos primeiro")