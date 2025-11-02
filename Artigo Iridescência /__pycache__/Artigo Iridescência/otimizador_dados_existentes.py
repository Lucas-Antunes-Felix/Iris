import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from simulacao import run_simulation
from graficos import analisar_cores_para_comprimento_onda
import matplotlib.pyplot as plt


class OtimizadorBlackFilmTransicao:
    def __init__(self, dados_temporais):
        """
        Otimizador que detecta black film em t=60s e depois otimiza espectros
        com restrições usando Nelder-Mead
        """
        self.dados_temporais = dados_temporais
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # Preparar dados para comparação
        self.posicoes_comparacao = np.linspace(0, 4, 500)

        # Pré-processar interpolações
        self.interpoladores_reais = {}
        for tempo, dados in dados_temporais.items():
            interp_real = interp1d(
                dados['dados_completos']['posicoes_cm'],
                dados['dados_completos']['comprimentos_onda_nm'],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            self.interpoladores_reais[tempo] = interp_real

    def verificar_black_film_tempo(self, h0, alpha, beta, tempo, limite_variacao=2.0):
        """
        Verifica se em determinado tempo está em BLACK FILM (True) ou com CORES (False)
        ANALISANDO REGIÃO 0-0.5cm (BEM perto do menisco)
        """
        params_sim = {
            'h0': h0, 'alpha': alpha, 'beta': beta,
            'n_film': 1.375, 'num_steps': 1000, 't_initial': tempo
        }

        try:
            resultados_sim = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'], resultados_sim['x_cm']
            )

            # ⚠️ CORREÇÃO: Filtrar região 0-0.5cm (BEM perto do menisco)
            posicoes = analise_sim['posicoes_cm']
            comprimentos_onda = analise_sim['comprimentos_onda_nm']
            mascara_regiao_0 = posicoes <= 0.5  # ← 0-0.5cm

            if len(comprimentos_regiao_0) < 5:
                return True  # Assume black film se poucos dados

            # Calcular VARIÂNCIA apenas na região 0-0.5cm
            variância = np.var(comprimentos_regiao_0)

            # Variância BAIXA = BLACK FILM REAL (sem cores perto do menisco)
            return variância < limite_variacao  # ← Usando 2.0 como limite

        except Exception as e:
            return True  # Assume black film em caso de erro

    def verificar_consistencia_temporal(self, h0, alpha, beta):
        """
        Verifica a consistência temporal dos parâmetros:
        - t=60s: DEVE estar em BLACK FILM (variância baixa)
        - t=70,80,90s: NÃO deve estar em black film (variância alta)
        """
        # t=60s: DEVE estar em BLACK FILM
        black_film_60s = self.verificar_black_film_tempo(h0, alpha, beta, 60)

        # t=70,80,90s: NÃO devem estar em BLACK FILM
        cores_70s = not self.verificar_black_film_tempo(h0, alpha, beta, 70)
        cores_80s = not self.verificar_black_film_tempo(h0, alpha, beta, 80)
        cores_90s = not self.verificar_black_film_tempo(h0, alpha, beta, 90)

        return black_film_60s, cores_70s, cores_80s, cores_90s

    def detectar_parametros_criticos(self, h0_inicial=2700e-9, alpha_inicial=0.02, beta_inicial=2e-08):
        """
        FASE 1: Detecta os parâmetros onde black film começa EM t=60s
        ANALISANDO REGIÃO 0-0.5cm (BEM perto do menisco)
        """
        print("🎯 FASE 1: DETECTANDO PARÂMETROS CRÍTICOS (Black film em t=60s)")
        print("   ⚠️  ANALISANDO REGIÃO 0-0.5cm (BEM perto do menisco)")
        print("=" * 60)

        h0_atual = h0_inicial
        alpha_atual = alpha_inicial
        beta_atual = beta_inicial

        # Redução gradual dos parâmetros
        fator_reducao = 0.95  # -5% a cada iteração
        fator_aumento_alpha = 1.03  # +3% alpha
        fator_aumento_beta = 1.02  # +2% beta

        for iteracao in range(50):  # Máximo 50 iterações
            # Simular para t=60s
            params_sim = {
                'h0': h0_atual, 'alpha': alpha_atual, 'beta': beta_atual,
                'n_film': 1.375, 'num_steps': 1000, 't_initial': 60
            }

            try:
                resultados_sim = run_simulation(params_sim)
                analise_sim = analisar_cores_para_comprimento_onda(
                    resultados_sim['colors_rgb'], resultados_sim['x_cm']
                )

                # ⚠️ CORREÇÃO CRÍTICA: Filtrar região 0-0.5cm (BEM perto do menisco)
                posicoes = analise_sim['posicoes_cm']
                comprimentos_onda = analise_sim['comprimentos_onda_nm']
                mascara_regiao_0 = posicoes <= 0.5  # ← 0-0.5cm EM VEZ DE 0-2cm

                if len(comprimentos_regiao_0) > 5:  # Menos pontos são suficientes
                    variância = np.var(comprimentos_regiao_0)

                    print(f"   Iter {iteracao + 1}: h0={h0_atual * 1e9:.1f}nm, var={variância:.1f} (0-0.5cm)")

                    # ⚠️ CRITÉRIO MAIS RESTRITIVO: variância BEM baixa = black film REAL
                    if variância < 2.0:  # ← Limite mais baixo para black film real
                        print(f"   ✅ BLACK FILM DETECTADO BEM PRÓXIMO AO MENISCO!")
                        print(
                            f"      Parâmetros críticos: h0={h0_atual * 1e9:.1f}nm, α={alpha_atual:.3e}, β={beta_atual:.3e}")
                        return h0_atual, alpha_atual, beta_atual

            except Exception as e:
                pass

            # Reduzir parâmetros gradualmente
            h0_atual *= fator_reducao
            alpha_atual *= fator_aumento_alpha
            beta_atual *= fator_aumento_beta

            # Parâmetros mínimos físicos
            if h0_atual < 100e-9:
                h0_atual = 100e-9
            if alpha_atual > 0.5:
                alpha_atual = 0.5
            if beta_atual > 1e-06:
                beta_atual = 1e-06

        print("   ⚠️  Black film não detectado na região 0-0.5cm, usando valores iniciais")
        return h0_inicial, alpha_inicial, beta_inicial

    def calcular_erro_espectral_tempo(self, h0, alpha, beta, tempo):
        """Calcula erro espectral para um tempo específico"""
        try:
            params_sim = {
                'h0': h0, 'alpha': alpha, 'beta': beta,
                'n_film': 1.375, 'num_steps': 1000, 't_initial': tempo
            }

            resultados_sim = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'], resultados_sim['x_cm']
            )

            # Interpolar simulação para mesma escala
            interp_sim = interp1d(
                analise_sim['posicoes_cm'],
                analise_sim['comprimentos_onda_nm'],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            comp_sim_interp = interp_sim(self.posicoes_comparacao)

            # Dados reais interpolados
            interp_real = self.interpoladores_reais[tempo]
            comp_real_interp = interp_real(self.posicoes_comparacao)

            # Calcular RMSE
            valid_mask = ~np.isnan(comp_sim_interp) & ~np.isnan(comp_real_interp)
            if np.sum(valid_mask) > 10:
                return np.sqrt(np.mean((comp_sim_interp[valid_mask] - comp_real_interp[valid_mask]) ** 2))
            else:
                return 1000

        except Exception as e:
            return 1000

    def funcao_custo_nelder_mead(self, params_otimizacao, params_criticos):
        """
        Função custo para Nelder-Mead com restrições
        """
        h0, alpha, beta = params_otimizacao
        h0_critico, alpha_critico, beta_critico = params_criticos

        # 1. CALCULAR ERRO ESPECTRAL NOS 4 TEMPOS
        erro_espectral = 0
        for tempo in [60, 70, 80, 90]:
            erro_tempo = self.calcular_erro_espectral_tempo(h0, alpha, beta, tempo)
            erro_espectral += erro_tempo

        # 2. APLICAR RESTRIÇÕES VIA PENALIDADES
        penalidade = 0

        # Restrição: h0 ≥ h0_critico (não pode ser menor que o valor crítico)
        if h0 < h0_critico:
            penalidade += 1000 * (h0_critico - h0) / h0_critico
            print(f"      ⚠️  RESTRIÇÃO h0: {h0 * 1e9:.1f}nm < {h0_critico * 1e9:.1f}nm")

        # Restrição: alpha ≥ alpha_critico
        if alpha < alpha_critico:
            penalidade += 500 * (alpha_critico - alpha) / alpha_critico
            print(f"      ⚠️  RESTRIÇÃO alpha: {alpha:.3e} < {alpha_critico:.3e}")

        # Restrição: beta ≥ beta_critico
        if beta < beta_critico:
            penalidade += 500 * (beta_critico - beta) / beta_critico
            print(f"      ⚠️  RESTRIÇÃO beta: {beta:.3e} < {beta_critico:.3e}")

        # Penalidade para parâmetros fisicamente impossíveis
        if h0 <= 0 or alpha <= 0 or beta <= 0:
            penalidade += 1000
            print(f"      ❌ Parâmetros impossíveis!")

        # 3. COMBINAR ERROS
        erro_total = erro_espectral + penalidade

        # 4. ATUALIZAR MELHOR RESULTADO
        if erro_total < self.melhor_erro:
            self.melhor_erro = erro_total
            self.melhor_params = params_otimizacao.copy()

        # 5. MOSTRAR PROGRESSO
        self.iteracao_atual += 1
        indicador = "✨" if erro_total < self.melhor_erro else "➡️"

        print(
            f"   [{self.iteracao_atual:2d}] {indicador} Erro: {erro_total:.1f} = {erro_espectral:.1f} (esp) + {penalidade:.1f} (pen)")
        print(f"      h0={h0 * 1e9:.1f}nm | α={alpha:.3e} | β={beta:.3e}")
        print("-" * 50)

        return erro_total

    def otimizar_com_restricoes(self, h0_inicial=2700e-9, alpha_inicial=0.02, beta_inicial=2e-08, max_iter=5):
        """
        Estratégia completa com detecção + otimização com restrições
        """
        print("🎯 INICIANDO OTIMIZAÇÃO COM RESTRIÇÕES")
        print("=" * 80)

        # FASE 1: Detectar parâmetros críticos
        h0_critico, alpha_critico, beta_critico = self.detectar_parametros_criticos(
            h0_inicial, alpha_inicial, beta_inicial
        )

        print(f"\n📊 PARÂMETROS CRÍTICOS ENCONTRADOS:")
        print(f"   h0: {h0_critico * 1e9:.1f} nm")
        print(f"   alpha: {alpha_critico:.3e}")
        print(f"   beta: {beta_critico:.3e}")
        print("-" * 60)

        # FASE 2: Otimização Nelder-Mead com restrições
        print("🎯 FASE 2: OTIMIZAÇÃO NELDER-MEAD COM RESTRIÇÕES")
        print("   Restrições: h0, alpha, beta ≥ valores críticos")
        print("-" * 60)

        # Resetar contadores
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')

        # Ponto inicial: parâmetros críticos
        x0 = [h0_critico, alpha_critico, beta_critico]

        # Função custo com restrições
        def custo_com_restricoes(params):
            return self.funcao_custo_nelder_mead(params, (h0_critico, alpha_critico, beta_critico))

        # Executar Nelder-Mead
        resultado = minimize(
            custo_com_restricoes,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'disp': True,
                'xatol': 1e-6,
                'fatol': 1e-4,
                'adaptive': True
            }
        )

        h0_opt, alpha_opt, beta_opt = resultado.x

        print("=" * 80)
        print("✅ OTIMIZAÇÃO CONCLUÍDA!")
        print(f"   Iterações totais: {self.iteracao_atual}")
        print(f"   h0: {h0_inicial * 1e9:.1f} → {h0_opt * 1e9:.1f} nm")
        print(f"   alpha: {alpha_inicial:.3e} → {alpha_opt:.3e}")
        print(f"   beta: {beta_inicial:.3e} → {beta_opt:.3e}")
        print(f"   Erro final: {resultado.fun:.2f}")

        # Verificar se restrições foram respeitadas
        print(f"\n📊 VERIFICAÇÃO DAS RESTRIÇÕES:")
        print(f"   h0: {h0_opt * 1e9:.1f}nm ≥ {h0_critico * 1e9:.1f}nm → {'✅' if h0_opt >= h0_critico else '❌'}")
        print(f"   alpha: {alpha_opt:.3e} ≥ {alpha_critico:.3e} → {'✅' if alpha_opt >= alpha_critico else '❌'}")
        print(f"   beta: {beta_opt:.3e} ≥ {beta_critico:.3e} → {'✅' if beta_opt >= beta_critico else '❌'}")

        # Parâmetros finais
        params_otimizados = {
            'h0': max(h0_opt, 100e-9),
            'alpha': max(alpha_opt, 1e-6),
            'beta': max(beta_opt, 1e-12),
            'n_film': 1.375
        }

        return {
            'params_otimizados': params_otimizados,
            'params_criticos': {
                'h0': h0_critico, 'alpha': alpha_critico, 'beta': beta_critico
            },
            'resultado_otimizacao': resultado,
            'erro_final': resultado.fun,
            'restricoes_respeitadas': (
                    h0_opt >= h0_critico and
                    alpha_opt >= alpha_critico and
                    beta_opt >= beta_critico
            )
        }

    def plotar_resultados_finais(self, resultado_otimizacao):
        """
        Plota os 4 gráficos de comparação final
        """
        params_opt = resultado_otimizacao['params_otimizados']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        tempos = [60, 70, 80, 90]

        for idx, tempo in enumerate(tempos):
            # Simular com parâmetros OTIMIZADOS
            params_sim = {
                'h0': params_opt['h0'],
                'alpha': params_opt['alpha'],
                'beta': params_opt['beta'],
                'n_film': 1.375,
                'num_steps': 1000,
                't_initial': tempo
            }

            simulacao = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                simulacao['colors_rgb'],
                simulacao['x_cm']
            )

            # Dados reais
            dados_reais = self.dados_temporais[tempo]

            # Plot
            ax = axes[idx]

            ax.plot(dados_reais['dados_completos']['posicoes_cm'],
                    dados_reais['dados_completos']['comprimentos_onda_nm'],
                    'ko-', linewidth=2, markersize=3, label='Experimental', alpha=0.8)

            ax.plot(analise_sim['posicoes_cm'], analise_sim['comprimentos_onda_nm'],
                    'r-', linewidth=2, label='Simulado', alpha=0.8)

            ax.set_xlabel('Posição (cm)')
            ax.set_ylabel('Comprimento de Onda (nm)')
            ax.set_title(f't = {tempo}s\nh₀ = {params_opt["h0"] * 1e9:.1f} nm')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 4)
            ax.set_ylim(400, 700)

        plt.tight_layout()
        plt.show()


def executar_otimizacao_inteligente(dados_temporais, max_iter=30, h0_inicial=2700e-9, alpha_inicial=0.02,
                                    beta_inicial=2e-08):
    """
    Função principal para otimização com restrições
    """
    print(f"🎯 OTIMIZADOR COM DETECÇÃO DE BLACK FILM + RESTRIÇÕES")

    otimizador = OtimizadorBlackFilmTransicao(dados_temporais)

    resultado = otimizador.otimizar_com_restricoes(
        max_iter=max_iter,
        h0_inicial=h0_inicial,
        alpha_inicial=alpha_inicial,
        beta_inicial=beta_inicial
    )

    # Plotar resultados
    otimizador.plotar_resultados_finais(resultado)

    return resultado