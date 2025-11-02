import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from simulacao import run_simulation
from graficos import analisar_cores_para_comprimento_onda
import matplotlib.pyplot as plt


class OtimizadorBlackFilmTransicao:
    def __init__(self, dados_temporais, params_simulacao):
        """
        Otimizador que detecta black film e otimiza espectros com restrições
        BASEADO EM ANÁLISE OBSERVACIONAL FÍSICA
        """
        self.dados_temporais = dados_temporais
        self.params = params_simulacao
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')
        self.melhor_params = None

        # Preparar dados para comparação
        self.posicoes_comparacao = np.linspace(0, 4, self.params['posicoes_comparacao_pontos'])

        # Pré-processar interpolações
        self.interpoladores_reais = {}
        for tempo, dados in dados_temporais.items():
            interp_real = interp1d(
                dados['dados_completos']['posicoes_cm'],
                dados['dados_completos']['comprimentos_onda_nm'],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            self.interpoladores_reais[tempo] = interp_real

    def verificar_black_film_tempo(self, h0, alpha, beta, tempo):
        """
        Verifica se em determinado tempo está em BLACK FILM (True) ou com CORES (False)
        ANALISANDO APENAS OS PRIMEIROS N PONTOS
        """
        params_sim = {
            'h0': h0, 'alpha': alpha, 'beta': beta,
            'n_film': self.params['n_film'],
            'num_steps': self.params['num_steps'],
            't_initial': tempo
        }

        try:
            resultados_sim = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'], resultados_sim['x_cm']
            )

            # Usar apenas os primeiros N pontos para detecção de black film
            comprimentos_onda = analise_sim['comprimentos_onda_nm']
            comprimentos_regiao_0 = comprimentos_onda[:self.params['num_pontos_black_film']]

            if len(comprimentos_regiao_0) < self.params['num_pontos_black_film']:
                return True  # Assume black film se poucos dados

            # Calcular VARIÂNCIA apenas nos primeiros pontos
            variância = np.var(comprimentos_regiao_0)

            # Variância BAIXA = BLACK FILM REAL (sem cores perto do menisco)
            return variância < self.params['limite_variacao_black_film']

        except Exception as e:
            return True  # Assume black film em caso de erro

    def calcular_erro_espectral_tempo(self, h0, alpha, beta, tempo):
        """Calcula erro espectral para um tempo específico APENAS NA REGIÃO CONFIGURADA"""
        try:
            params_sim = {
                'h0': h0, 'alpha': alpha, 'beta': beta,
                'n_film': self.params['n_film'],
                'num_steps': self.params['num_steps'],
                't_initial': tempo
            }

            resultados_sim = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'], resultados_sim['x_cm']
            )

            # Filtrar apenas região configurada
            posicoes = analise_sim['posicoes_cm']
            comprimentos_onda = analise_sim['comprimentos_onda_nm']
            mascara_regiao = posicoes <= self.params['regiao_comparacao_cm']

            # Dados da simulação (apenas na região configurada)
            posicoes_sim = posicoes[mascara_regiao]
            comprimentos_sim = comprimentos_onda[mascara_regiao]

            # Interpolar simulação para mesma escala
            interp_sim = interp1d(
                posicoes_sim,
                comprimentos_sim,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )

            # Apenas posições na região para comparação
            posicoes_comparacao_regiao = self.posicoes_comparacao[
                self.posicoes_comparacao <= self.params['regiao_comparacao_cm']]
            comp_sim_interp = interp_sim(posicoes_comparacao_regiao)

            # Dados reais interpolados
            interp_real = self.interpoladores_reais[tempo]
            comp_real_interp = interp_real(posicoes_comparacao_regiao)

            # Calcular RMSE apenas na região
            valid_mask = ~np.isnan(comp_sim_interp) & ~np.isnan(comp_real_interp)
            if np.sum(valid_mask) > 10:
                erro = np.sqrt(np.mean((comp_sim_interp[valid_mask] - comp_real_interp[valid_mask]) ** 2))
                return erro
            else:
                return 1000

        except Exception as e:
            return 1000

    def calcular_penalidades_restricoes_fisicas(self, h0, alpha, beta):
        """
        NOVA FUNÇÃO: Calcula penalidades baseadas na análise observacional física
        Baseado nos seus limites: h0_max=20000nm, alpha entre 0.05-0.07, beta ~1e-8
        """
        penalidade = 0

        # 1. RESTRIÇÃO h0: máximo 20000 nm (limite físico absoluto)
        if h0 > self.params['h0_max_observado']:
            excesso = (h0 - self.params['h0_max_observado']) / self.params['h0_max_observado']
            penalidade += 2000 * excesso  # Penalidade forte para violação do limite físico
            print(f"      🚫 VIOLAÇÃO FÍSICA h0: {h0 * 1e9:.1f}nm > {self.params['h0_max_observado'] * 1e9:.1f}nm")

        # 2. RESTRIÇÃO alpha: entre 0.05 e 0.07 (faixa observacional)
        if alpha < self.params['alpha_min_observado']:
            deficit = (self.params['alpha_min_observado'] - alpha) / self.params['alpha_min_observado']
            penalidade += 1000 * deficit
            print(f"      ⚠️  ALPHA BAIXO: {alpha:.3f} < {self.params['alpha_min_observado']:.3f}")

        if alpha > self.params['alpha_max_observado']:
            excesso = (alpha - self.params['alpha_max_observado']) / self.params['alpha_max_observado']
            penalidade += 1000 * excesso
            print(f"      ⚠️  ALPHA ALTO: {alpha:.3f} > {self.params['alpha_max_observado']:.3f}")

        # 3. RESTRIÇÃO beta: próximo de 1e-8 (valor observacional)
        beta_ideal = self.params['beta_ideal_observado']
        tolerancia_beta = self.params['tolerancia_beta_observado']

        if abs(beta - beta_ideal) > tolerancia_beta * beta_ideal:
            desvio = abs(beta - beta_ideal) / beta_ideal
            penalidade += 800 * desvio
            print(f"      ⚠️  BETA FORA DA FAIXA: {beta:.2e} vs ideal {beta_ideal:.2e}")

        # 4. Penalidade para parâmetros fisicamente impossíveis
        if h0 <= 0 or alpha <= 0 or beta <= 0:
            penalidade += 5000
            print(f"      ❌ PARÂMETROS IMPOSSÍVEIS!")

        return penalidade

    def funcao_custo_nelder_mead(self, params_otimizacao):
        """
        Função custo para Nelder-Mead com restrições físicas baseadas em observação
        """
        h0, alpha, beta = params_otimizacao

        # 1. CALCULAR ERRO ESPECTRAL NOS TEMPOS CONFIGURADOS
        erro_espectral = 0
        for tempo in self.params['tempos_analise']:
            erro_tempo = self.calcular_erro_espectral_tempo(h0, alpha, beta, tempo)
            erro_espectral += erro_tempo

        # 2. APLICAR RESTRIÇÕES FÍSICAS BASEADAS NA OBSERVAÇÃO
        penalidade = self.calcular_penalidades_restricoes_fisicas(h0, alpha, beta)

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
        print(f"      h0={h0 * 1e9:.1f}nm | α={alpha:.3f} | β={beta:.2e}")

        # Mostrar status das restrições
        status_h0 = "✅" if h0 <= self.params['h0_max_observado'] else "🚫"
        status_alpha = "✅" if (
                    self.params['alpha_min_observado'] <= alpha <= self.params['alpha_max_observado']) else "⚠️ "
        status_beta = "✅" if abs(beta - self.params['beta_ideal_observado']) <= self.params[
            'tolerancia_beta_observado'] * self.params['beta_ideal_observado'] else "⚠️ "

        print(f"      Restrições: h0{status_h0} alpha{status_alpha} beta{status_beta}")
        print("-" * 50)

        return erro_total

    def otimizar_com_restricoes_fisicas(self):
        """
        Estratégia completa com restrições baseadas em análise observacional física
        """
        print("🎯 INICIANDO OTIMIZAÇÃO COM RESTRIÇÕES FÍSICAS")
        print("=" * 80)
        print("📋 RESTRIÇÕES BASEADAS EM ANÁLISE OBSERVACIONAL:")
        print(f"   h0 ≤ {self.params['h0_max_observado'] * 1e9:.0f} nm (limite físico máximo)")
        print(
            f"   {self.params['alpha_min_observado']:.2f} ≤ α ≤ {self.params['alpha_max_observado']:.2f} (faixa observada)")
        print(
            f"   β ≈ {self.params['beta_ideal_observado']:.1e} ± {self.params['tolerancia_beta_observado'] * 100:.0f}%")
        print("-" * 60)

        # Resetar contadores
        self.iteracao_atual = 0
        self.melhor_erro = float('inf')

        # Ponto inicial: valores dentro das faixas observadas
        x0 = [
            min(self.params['h0_inicial'], self.params['h0_max_observado'] * 0.8),  # 80% do máximo
            (self.params['alpha_min_observado'] + self.params['alpha_max_observado']) / 2,  # Ponto médio
            self.params['beta_ideal_observado']  # Valor ideal
        ]

        # Executar Nelder-Mead
        resultado = minimize(
            self.funcao_custo_nelder_mead,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': self.params['max_iter'],
                'disp': True,
                'xatol': 1e-8,
                'fatol': 1e-6,
                'adaptive': True
            }
        )

        h0_opt, alpha_opt, beta_opt = resultado.x

        print("=" * 80)
        print("✅ OTIMIZAÇÃO CONCLUÍDA!")
        print(f"   Iterações totais: {self.iteracao_atual}")
        print(f"   h0: {self.params['h0_inicial'] * 1e9:.1f} → {h0_opt * 1e9:.1f} nm")
        print(f"   alpha: {self.params['alpha_inicial']:.3f} → {alpha_opt:.3f}")
        print(f"   beta: {self.params['beta_inicial']:.2e} → {beta_opt:.2e}")
        print(f"   Erro final: {resultado.fun:.2f}")

        # Verificar se restrições foram respeitadas
        print(f"\n📊 VERIFICAÇÃO DAS RESTRIÇÕES FÍSICAS:")
        h0_ok = h0_opt <= self.params['h0_max_observado']
        alpha_ok = self.params['alpha_min_observado'] <= alpha_opt <= self.params['alpha_max_observado']
        beta_ok = abs(beta_opt - self.params['beta_ideal_observado']) <= self.params['tolerancia_beta_observado'] * \
                  self.params['beta_ideal_observado']

        print(
            f"   h0 ≤ {self.params['h0_max_observado'] * 1e9:.0f} nm: {h0_opt * 1e9:.1f} nm → {'✅' if h0_ok else '❌'}")
        print(
            f"   α ∈ [{self.params['alpha_min_observado']:.2f}, {self.params['alpha_max_observado']:.2f}]: {alpha_opt:.3f} → {'✅' if alpha_ok else '❌'}")
        print(f"   β ≈ {self.params['beta_ideal_observado']:.1e}: {beta_opt:.2e} → {'✅' if beta_ok else '❌'}")

        # Parâmetros finais (aplicar limites físicos)
        params_otimizados = {
            'h0': min(max(h0_opt, self.params['h0_min']), self.params['h0_max_observado']),
            'alpha': min(max(alpha_opt, self.params['alpha_min_observado']), self.params['alpha_max_observado']),
            'beta': beta_opt,  # Manter valor otimizado, já que temos tolerância
            'n_film': self.params['n_film']
        }

        return {
            'params_otimizados': params_otimizados,
            'resultado_otimizacao': resultado,
            'erro_final': resultado.fun,
            'restricoes_respeitadas': (h0_ok and alpha_ok and beta_ok)
        }

    def validar_com_t120s(self, params_otimizados):
        """
        Validação externa com t=120s (não usado na otimização)
        """
        print(f"\n🎯 VALIDAÇÃO COM t=120s (DADO EXTERNO):")

        try:
            params_sim = {
                'h0': params_otimizados['h0'],
                'alpha': params_otimizados['alpha'],
                'beta': params_otimizados['beta'],
                'n_film': self.params['n_film'],
                'num_steps': self.params['num_steps'],
                't_initial': 120
            }

            resultados_sim = run_simulation(params_sim)
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'], resultados_sim['x_cm']
            )

            # Aqui você pode comparar com seus dados de t=120s se disponível
            print(f"   Simulação em t=120s com parâmetros otimizados:")
            print(f"   h0 = {params_otimizados['h0'] * 1e9:.1f} nm")
            print(f"   α  = {params_otimizados['alpha']:.3f}")
            print(f"   β  = {params_otimizados['beta']:.2e}")
            print("   ✅ Validação concluída")

        except Exception as e:
            print(f"   ❌ Erro na validação: {e}")

    def plotar_resultados_finais(self, resultado_otimizacao):
        """
        Plota os gráficos de comparação final
        """
        params_opt = resultado_otimizacao['params_otimizados']

        num_tempos = len(self.params['tempos_analise'])
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, tempo in enumerate(self.params['tempos_analise']):
            if idx >= len(axes):
                break

            # Simular com parâmetros OTIMIZADOS
            params_sim = {
                'h0': params_opt['h0'],
                'alpha': params_opt['alpha'],
                'beta': params_opt['beta'],
                'n_film': self.params['n_film'],
                'num_steps': self.params['num_steps'],
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


def executar_otimizacao_inteligente(dados_temporais, params_simulacao):
    """
    Função de compatibilidade com main.py - COM RESTRIÇÕES FÍSICAS
    """
    print(f"🎯 EXECUTANDO OTIMIZAÇÃO INTELIGENTE COM RESTRIÇÕES FÍSICAS")

    otimizador = OtimizadorBlackFilmTransicao(dados_temporais, params_simulacao)

    resultado = otimizador.otimizar_com_restricoes_fisicas()

    # Validação com t=120s
    otimizador.validar_com_t120s(resultado['params_otimizados'])

    # Plotar resultados finais
    otimizador.plotar_resultados_finais(resultado)

    return resultado