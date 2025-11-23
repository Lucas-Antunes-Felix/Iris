import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import minimize

try:
    from simulacao import run_simulation
    from graficos import analisar_cores_para_comprimento_onda
except ImportError as e:
    print(f"⚠️  Aviso de importação: {e}")


class OtimizadorInteligente:
    def __init__(self, dados_reais, params_simulacao):
        self.dados_reais = dados_reais
        self.params = params_simulacao
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

    def simular_com_parametros(self, h0, alpha, beta, tempos=None):
        """Simulação para todos os tempos"""
        if tempos is None:
            tempos = [60, 70, 80, 90]

        dados_simulados = {}
        for tempo in tempos:
            try:
                params_sim = {
                    'h0': h0, 'alpha': alpha, 'beta': beta,
                    'n_film': self.params['n_film'],
                    'num_steps': 500,
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
                print(f"❌ Erro simulação t={tempo}s: {e}")
                continue

        return dados_simulados

    def calcular_erro_simplificado(self, dados_reais_formatados, dados_simulados):
        """
        🎯 FUNÇÃO CUSTO SIMPLIFICADA - Evita importação circular
        """
        try:
            todas_diferencas = []

            for tempo in [60, 70, 80, 90]:
                if tempo not in dados_reais_formatados or tempo not in dados_simulados:
                    continue

                # Calcular diferença entre comprimentos de onda médios
                lambda_medio_real = np.mean(dados_reais_formatados[tempo]['comprimentos_onda_nm'])
                lambda_medio_sim = np.mean(dados_simulados[tempo]['comprimentos_onda_nm'])

                diferenca = abs(lambda_medio_real - lambda_medio_sim)
                todas_diferencas.append(diferenca)

                # Penalizar se não há dados suficientes
                if len(dados_reais_formatados[tempo]['comprimentos_onda_nm']) < 10:
                    todas_diferencas.append(100)  # Penalidade alta

            if not todas_diferencas:
                return 5000

            # Erro médio em nm, convertido para escala de custo
            erro_medio = np.mean(todas_diferencas)
            erro_total = erro_medio * 10  # Fator de escala

            return min(erro_total, 5000)  # Limitar erro máximo

        except Exception as e:
            print(f"      ❌ Erro cálculo simplificado: {e}")
            return 5000

    def calcular_erro_inteligente(self, h0, alpha, beta):
        """
        🎯 FUNÇÃO CUSTO INTELIGENTE - Versão Corrigida
        """
        try:
            # 1. Simular
            dados_simulados = self.simular_com_parametros(h0, alpha, beta)

            # 2. Preparar dados reais
            dados_reais_formatados = {}
            for tempo, dados in self.dados_reais.items():
                if 'dados_completos' in dados:
                    dados_reais_formatados[tempo] = {
                        'posicoes_cm': dados['dados_completos']['posicoes_cm'],
                        'comprimentos_onda_nm': dados['dados_completos']['comprimentos_onda_nm']
                    }

            # 3. 🎯 USAR MÉTODO SIMPLIFICADO (evita importação circular)
            erro = self.calcular_erro_simplificado(dados_reais_formatados, dados_simulados)

            # Log resumido
            if self.iteracao_atual % 5 == 0:
                print(f"      🔍 It {self.iteracao_atual}: erro={erro:.1f}")

            return erro

        except Exception as e:
            print(f"      ❌ Erro cálculo inteligente: {e}")
            return 5000

    def calcular_erro_grafico4(self, h0, alpha, beta):
        """
        🎯 FUNÇÃO CUSTO ESPECÍFICA PARA GRÁFICO 4 - CORRIGIDA
        """
        try:
            print(f"      📊 Calculando erro Gráfico4...")

            # 1. Simular com parâmetros atuais
            dados_simulados = self.simular_com_parametros(h0, alpha, beta)

            # 2. Executar análise do Gráfico 4
            from incerteza import AnalisadorIncertezas

            # 🎯 CORREÇÃO: Criar params atualizados
            params_atualizados = {
                'h0': h0,
                'alpha': alpha,
                'beta': beta,
                'n_film': self.params['n_film']
            }

            analisador = AnalisadorIncertezas(self.dados_reais, dados_simulados, params_atualizados)

            # 🎯 CORREÇÃO: Aplicar shift se o método existir
            try:
                if hasattr(analisador, 'aplicar_shift_automatico'):
                    dados_simulados_com_shift = analisador.aplicar_shift_automatico(self.dados_reais, dados_simulados)
                    analisador.dados_simulados = dados_simulados_com_shift
                else:
                    # Método não existe, usar dados originais
                    analisador.dados_simulados = dados_simulados
                    print("      ⚠️  Método aplicar_shift_automatico não disponível")
            except Exception as e:
                print(f"      ⚠️  Erro no shift: {e}")
                analisador.dados_simulados = dados_simulados

            # Analisar todos os tempos
            resultados_analise = analisador.analisar_todos_tempos()

            if not resultados_analise:
                print("      ❌ Análise Gráfico4 falhou")
                return 10000

            # ... resto do código original (coletar diferenças, calcular erro)
            todas_diferencas = []
            penalidades_desaparecimento = 0
            recompensa_black_film_70 = 0
            total_cores_perdidas = 0

            for tempo in [60, 70, 80, 90]:
                if tempo not in resultados_analise:
                    continue

                resultado_tempo = resultados_analise[tempo]

                # Contar cores detectadas
                cores_reais = list(resultado_tempo['pontos_reais'].keys())
                cores_sim = list(resultado_tempo['pontos_simulados'].keys())

                print(f"      🎯 t={tempo}s: {len(cores_reais)} cores reais vs {len(cores_sim)} cores simuladas")

                # 🎯 PUNIÇÃO POR CORES PERDIDAS
                cores_perdidas = [cor for cor in cores_reais if cor not in cores_sim]
                if cores_perdidas:
                    penalidade = len(cores_perdidas) * 300
                    penalidades_desaparecimento += penalidade
                    total_cores_perdidas += len(cores_perdidas)
                    print(
                        f"      ❌ t={tempo}s: {len(cores_perdidas)} cores perdidas {cores_perdidas} → +{penalidade} penalidade")

                # 🎯 RECOMPENSA POR BLACK FILM EM t=70
                if tempo == 70 and 'black_film' in cores_sim:
                    recompensa_black_film_70 = -400
                    print(f"      🎉 BLACK FILM detectado em t=70s! → {recompensa_black_film_70} recompensa")

                # Coletar diferenças para cores que existem em ambos
                for cor, diff in resultado_tempo['diferencas_posicionais'].items():
                    if diff is not None:
                        erro_abs = abs(diff)
                        todas_diferencas.append(erro_abs)
                        print(f"      📏 {cor}: Δ={diff:.3f}cm")

            # 4. CALCULAR ERRO TOTAL
            if not todas_diferencas:
                erro_medio = 5.0
            else:
                erro_medio = np.mean(todas_diferencas)

            erro_principal = erro_medio * 150
            erro_total = (erro_principal + penalidades_desaparecimento + recompensa_black_film_70)
            erro_total = max(erro_total, 0)

            print(f"      📊 RESUMO GRÁFICO4:")
            print(f"         • Diferença média: {erro_medio:.3f}cm → {erro_principal:.1f}pts")
            print(f"         • Cores perdidas: {total_cores_perdidas} → +{penalidades_desaparecimento:.1f}pts")
            print(f"         • Recompensa black film t=70: {recompensa_black_film_70:.1f}pts")
            print(f"         • 🎯 ERRO TOTAL: {erro_total:.1f}pts")

            return erro_total

        except Exception as e:
            print(f"      ❌ Erro cálculo Gráfico4: {e}")
            import traceback
            traceback.print_exc()
            return 10000

    def funcao_custo_grafico4(self, params):
        """
        🎯 FUNÇÃO CUSTO PRINCIPAL - FOCO NO GRÁFICO 4
        """
        h0, alpha, beta = params

        # Restrições físicas
        if h0 <= 0 or alpha <= 0 or beta <= 0:
            return 10000

        # Calcular erro baseado no Gráfico 4
        erro = self.calcular_erro_grafico4(h0, alpha, beta)

        # Atualizar melhor resultado
        if erro < self.melhor_erro:
            self.melhor_erro = erro
            self.melhor_params = params.copy()
            print(f"      🎉 MELHORIA! Erro Gráfico4: {erro:.1f}")

        self.iteracao_atual += 1

        # Log compacto
        print(f"   [{self.iteracao_atual:2d}] Erro: {erro:.1f} | "
              f"h0={h0 * 1e9:.0f}nm α={alpha:.4f} β={beta:.2e}")

        return erro

    def funcao_custo(self, params):
        """Função custo original - para compatibilidade"""
        h0, alpha, beta = params

        # Restrições físicas
        if h0 <= 0 or alpha <= 0 or beta <= 0:
            return 10000

        # Calcular erro
        erro = self.calcular_erro_inteligente(h0, alpha, beta)

        # Atualizar melhor resultado
        if erro < self.melhor_erro:
            self.melhor_erro = erro
            self.melhor_params = params.copy()
            print(f"      🎉 MELHORIA! Erro: {erro:.1f}")

        self.iteracao_atual += 1

        # Log compacto
        print(f"   [{self.iteracao_atual:2d}] Erro: {erro:.1f} | "
              f"h0={h0 * 1e9:.0f}nm α={alpha:.4f} β={beta:.2e}")

        return erro

    def executar_otimizacao_grafico4(self):
        """
        🎯 OTIMIZAÇÃO ESPECÍFICA PARA MINIMIZAR ERROS DO GRÁFICO 4
        """
        print("🚀 INICIANDO OTIMIZAÇÃO GRÁFICO 4")
        print("=" * 60)
        print("ESTRATÉGIA: Minimizar diferenças posicionais + Punir desaparecimento")
        print("BÔNUS: Black film em t=70s | PUNIÇÃO: Cores perdidas")
        print("-" * 60)

        # Resetar contadores
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # Parâmetros iniciais
        x0 = [
            self.params['h0'] * np.random.uniform(0.8, 1.2),
            self.params['alpha'] * np.random.uniform(0.8, 1.2),
            self.params['beta'] * np.random.uniform(0.8, 1.2)
        ]

        print("PARÂMETROS INICIAIS:")
        print(f"   h0: {x0[0] * 1e9:.0f} nm")
        print(f"   α:  {x0[1]:.4f}")
        print(f"   β:  {x0[2]:.2e}")
        print("-" * 60)

        # Limites de busca
        bounds = [
            (1000e-9, 30000e-9),
            (0.01, 0.15),
            (1e-9, 1e-7)
        ]

        # Executar otimização com foco no Gráfico 4
        try:
            resultado = minimize(
                self.funcao_custo_grafico4,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 25,  # Focado e rápido
                    'disp': True,
                    'xatol': 1e-6,
                    'fatol': 50,  # Tolerância maior para erros complexos
                    'adaptive': True
                }
            )
        except Exception as e:
            print(f"⚠️  Otimização interrompida: {e}")
            resultado = None

        # 🎯 USAR MELHORES PARÂMETROS
        if self.melhor_params is not None:
            h0_opt, alpha_opt, beta_opt = self.melhor_params
            erro_final = self.melhor_erro
            melhorou = True
        else:
            h0_opt, alpha_opt, beta_opt = x0
            erro_final = self.melhor_erro
            melhorou = False

        # Resultados finais
        self._exibir_resultados_finais(
            self.params['h0'], self.params['alpha'], self.params['beta'],
            h0_opt, alpha_opt, beta_opt,
            erro_final
        )

        # Parâmetros otimizados
        params_otimizados = {
            'h0': max(h0_opt, 1000e-9),
            'alpha': max(alpha_opt, 0.001),
            'beta': max(beta_opt, 1e-12),
            'n_film': self.params['n_film']
        }

        return {
            'params_otimizados': params_otimizados,
            'resultado_otimizacao': resultado,
            'erro_final': erro_final,
            'melhorou': melhorou,
            'iteracoes_totais': self.iteracao_atual,
            'estrategia': 'grafico4'
        }

    def executar_otimizacao(self):
        """Otimização original - para compatibilidade"""
        print("🚀 INICIANDO OTIMIZAÇÃO INTELIGENTE")
        print("=" * 60)
        print("ESTRATÉGIA: Método simplificado (sem análise complexa)")
        print("-" * 60)

        # Resetar contadores
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # Parâmetros iniciais com variação
        x0 = [
            self.params['h0'] * np.random.uniform(0.8, 1.2),
            self.params['alpha'] * np.random.uniform(0.8, 1.2),
            self.params['beta'] * np.random.uniform(0.8, 1.2)
        ]

        print("PARÂMETROS INICIAIS (com variação):")
        print(f"   h0: {x0[0] * 1e9:.0f} nm")
        print(f"   α:  {x0[1]:.4f}")
        print(f"   β:  {x0[2]:.2e}")
        print("-" * 60)

        # Limites de busca
        bounds = [
            (1000e-9, 30000e-9),
            (0.01, 0.15),
            (1e-9, 1e-7)
        ]

        # Executar otimização
        try:
            resultado = minimize(
                self.funcao_custo,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 20,  # Reduzido para testes
                    'disp': True,
                    'xatol': 1e-6,
                    'fatol': 10,
                    'adaptive': True
                }
            )
        except Exception as e:
            print(f"⚠️  Otimização interrompida: {e}")
            resultado = None

        # 🎯 USAR MELHORES PARÂMETROS
        if self.melhor_params is not None:
            h0_opt, alpha_opt, beta_opt = self.melhor_params
            erro_final = self.melhor_erro
            melhorou = True
        else:
            h0_opt, alpha_opt, beta_opt = x0
            erro_final = self.melhor_erro
            melhorou = False

        # 🎯 EXIBIR RESULTADOS FINAIS
        self._exibir_resultados_finais(
            self.params['h0'], self.params['alpha'], self.params['beta'],
            h0_opt, alpha_opt, beta_opt,
            erro_final
        )

        # Parâmetros otimizados
        params_otimizados = {
            'h0': max(h0_opt, 1000e-9),
            'alpha': max(alpha_opt, 0.001),
            'beta': max(beta_opt, 1e-12),
            'n_film': self.params['n_film']
        }

        return {
            'params_otimizados': params_otimizados,
            'resultado_otimizacao': resultado,
            'erro_final': erro_final,
            'melhorou': melhorou,
            'iteracoes_totais': self.iteracao_atual
        }

    def _exibir_resultados_finais(self, h0_ini, alpha_ini, beta_ini,
                                  h0_opt, alpha_opt, beta_opt, erro_final):
        """Exibe resultados de forma clara"""
        print("\n" + "=" * 60)
        print("🎯 RESULTADOS FINAIS DA OTIMIZAÇÃO")
        print("=" * 60)

        print(f"📊 PARÂMETROS INICIAIS:")
        print(f"   • h₀ = {h0_ini * 1e9:.0f} nm")
        print(f"   • α  = {alpha_ini:.6f}")
        print(f"   • β  = {beta_ini:.2e}")

        print(f"\n✅ PARÂMETROS OTIMIZADOS:")
        print(f"   • h₀ = {h0_opt * 1e9:.0f} nm")
        print(f"   • α  = {alpha_opt:.6f}")
        print(f"   • β  = {beta_opt:.2e}")

        print(f"\n📈 ESTATÍSTICAS:")
        print(f"   • Iterações: {self.iteracao_atual}")
        print(f"   • Erro final: {erro_final:.1f}")

        # Calcular variações
        delta_h0 = ((h0_opt - h0_ini) / h0_ini * 100)
        delta_alpha = ((alpha_opt - alpha_ini) / alpha_ini * 100)
        delta_beta = ((beta_opt - beta_ini) / beta_ini * 100)

        print(f"\n📝 VARIAÇÕES:")
        print(f"   • Δh₀ = {delta_h0:+.1f}%")
        print(f"   • Δα  = {delta_alpha:+.1f}%")
        print(f"   • Δβ  = {delta_beta:+.1f}%")
        print("=" * 60)


# =============================================================================
# FUNÇÕES PRINCIPAIS (COMPATÍVEIS COM main.py)
# =============================================================================

def executar_otimizacao_inteligente(dados_reais, params_simulacao, estrategia='grafico4'):
    """
    Função principal - AGORA COM ESTRATÉGIA GRÁFICO 4
    """
    print("🚀 EXECUTANDO OTIMIZAÇÃO INTELIGENTE")
    print("=" * 60)

    if estrategia == 'grafico4':
        print("🎯 ESTRATÉGIA: Minimizar erros do GRÁFICO 4")
        print("   • Foco nas diferenças posicionais das cores")
        print("   • Punição severa por cores perdidas (+300 pts/cor)")
        print("   • Bônus por black film em t=70s (-400 pts)")
    else:
        print("📊 ESTRATÉGIA: Método simplificado (compatibilidade)")

    print("-" * 60)

    # Parâmetros antigos
    params_antigos = {
        'h0': params_simulacao['h0'],
        'alpha': params_simulacao['alpha'],
        'beta': params_simulacao['beta']
    }

    # Criar otimizador
    otimizador = OtimizadorInteligente(dados_reais, params_simulacao)

    # Executar otimização com estratégia escolhida
    if estrategia == 'grafico4':
        resultado_otimizacao = otimizador.executar_otimizacao_grafico4()
    else:
        resultado_otimizacao = otimizador.executar_otimizacao()

    # Salvar resultados
    salvar_resultados_simplificado(resultado_otimizacao, params_antigos)

    return resultado_otimizacao


def salvar_resultados_simplificado(resultado_otimizacao, params_antigos):
    """Salva resultados de forma simplificada"""
    params_opt = resultado_otimizacao['params_otimizados']

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
        'performance': {
            'erro_final': float(resultado_otimizacao['erro_final']),
            'melhorou': bool(resultado_otimizacao['melhorou']),
            'iteracoes_totais': int(resultado_otimizacao['iteracoes_totais'])
        }
    }

    # Adicionar estratégia se existir
    if 'estrategia' in resultado_otimizacao:
        dados_salvar['performance']['estrategia'] = resultado_otimizacao['estrategia']

    with open('resultados_otimizacao_simplificada.json', 'w', encoding='utf-8') as f:
        json.dump(dados_salvar, f, indent=2, ensure_ascii=False)

    print("💾 Resultados salvos em 'resultados_otimizacao_simplificada.json'")


# =============================================================================
# FUNÇÃO DE COMPATIBILIDADE
# =============================================================================

def analisar_erros_manual(params_manuais, params_simulacao):
    """
    Função para análise manual (compatibilidade) - CORRIGIDA
    """
    print("🔍 ANÁLISE MANUAL DE PARÂMETROS")
    print("=" * 50)

    # Carregar dados reais
    from analisador_foto import carregar_dados_existentes
    dados_reais = carregar_dados_existentes()

    if not dados_reais:
        print("❌ Dados reais não encontrados. Execute análise de fotos primeiro.")
        return None

    # Criar otimizador temporário
    otimizador = OtimizadorInteligente(dados_reais, params_simulacao)

    # Simular com parâmetros manuais
    dados_simulados = otimizador.simular_com_parametros(
        params_manuais['h0'],
        params_manuais['alpha'],
        params_manuais['beta']
    )

    # Calcular erro simples
    dados_reais_formatados = {}
    for tempo, dados in dados_reais.items():
        if 'dados_completos' in dados:
            dados_reais_formatados[tempo] = {
                'posicoes_cm': dados['dados_completos']['posicoes_cm'],
                'comprimentos_onda_nm': dados['dados_completos']['comprimentos_onda_nm']
            }

    erro = otimizador.calcular_erro_simplificado(dados_reais_formatados, dados_simulados)

    print(f"📊 Erro calculado: {erro:.1f}")

    return {
        'erro': erro,
        'dados_simulados': dados_simulados,
        'dados_reais': dados_reais_formatados
    }


if __name__ == "__main__":
    print("⚙️  OTIMIZADOR INTELIGENTE - VERSÃO GRÁFICO 4")
    print("=" * 60)
    print("CARACTERÍSTICAS:")
    print("   • 🎯 Estratégia Gráfico 4: diferenças posicionais")
    print("   • ❌ Punição: +300 pts por cor perdida")
    print("   • 🎉 Recompensa: -400 pts por black film em t=70")
    print("   • 📏 Foco: minimizar Δ posições no Gráfico 4")
    print("=" * 60)

    # Exemplo de uso
    from analisador_foto import carregar_dados_existentes

    dados = carregar_dados_existentes()
    if dados:
        print("✅ Dados carregados. Use executar_otimizacao_inteligente(dados, params_simulacao, estrategia='grafico4')")
    else:
        print("❌ Execute análise de fotos primeiro")