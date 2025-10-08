# [file name]: otimizador_dados.py
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from simulacao import run_simulation
import matplotlib.pyplot as plt


class OtimizadorDadosExistentes:
    def __init__(self, dados_reais, dados_simulados_iniciais, params_base):
        """
        Otimizador que usa dados já processados de comprimento de onda vs posição

        Args:
            dados_reais: dict com 'posicoes_cm' e 'comprimentos_onda_nm' da foto
            dados_simulados_iniciais: dict com 'posicoes_cm' e 'comprimentos_onda_nm' da simulação
            params_base: parâmetros base da simulação
        """
        self.dados_reais = dados_reais
        self.dados_sim_iniciais = dados_simulados_iniciais
        self.params_base = params_base.copy()

        # Preparar dados para comparação - interpolar para mesma escala
        self.posicoes_comparacao = np.linspace(0, 5, 500)  # Escala uniforme 0-5 cm

        # Interpolar dados reais
        self.interp_real = interp1d(
            dados_reais['posicoes_cm'],
            dados_reais['comprimentos_onda_nm'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self.comp_real_interp = self.interp_real(self.posicoes_comparacao)

        # Interpolar simulação inicial (para referência)
        self.interp_sim_inicial = interp1d(
            dados_simulados_iniciais['posicoes_cm'],
            dados_simulados_iniciais['comprimentos_onda_nm'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self.comp_sim_inicial_interp = self.interp_sim_inicial(self.posicoes_comparacao)

    def funcao_custo(self, params_otimizacao):
        """
        Função de custo: diferença entre simulação e dados reais
        """
        alpha, beta = params_otimizacao

        # Atualizar parâmetros
        self.params_base['alpha'] = alpha
        self.params_base['beta'] = beta

        try:
            # Executar simulação com novos parâmetros
            resultados_sim = run_simulation(self.params_base)

            # Converter cores da simulação para comprimentos de onda (usando sua função existente)
            from graficos import analisar_cores_para_comprimento_onda
            analise_sim = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'],
                resultados_sim['x_cm']
            )

            # Interpolar simulação atual para mesma escala de comparação
            interp_sim_atual = interp1d(
                analise_sim['posicoes_cm'],
                analise_sim['comprimentos_onda_nm'],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            comp_sim_atual_interp = interp_sim_atual(self.posicoes_comparacao)

            # Calcular erro (RMSE)
            valid_mask = ~np.isnan(comp_sim_atual_interp) & ~np.isnan(self.comp_real_interp)
            if np.sum(valid_mask) > 10:
                erro = np.sqrt(np.mean((comp_sim_atual_interp[valid_mask] - self.comp_real_interp[valid_mask]) ** 2))
            else:
                erro = 1000

            # Penalizar parâmetros fisicamente impossíveis
            if alpha <= 0 or beta <= 0:
                erro += 1000

            return erro

        except Exception as e:
            print(f"Erro na simulação: {e}")
            return 1000 + abs(alpha) + abs(beta)

    def otimizar(self, alpha_inicial=None, beta_inicial=None, metodo='Nelder-Mead', max_iter=50):
        """
        Executa a otimização para encontrar alpha e beta ótimos
        """
        print("🔧 Iniciando otimização de parâmetros...")

        # Usar valores iniciais dos parâmetros base se não especificados
        if alpha_inicial is None:
            alpha_inicial = self.params_base['alpha']
        if beta_inicial is None:
            beta_inicial = self.params_base['beta']

        x0 = [alpha_inicial, beta_inicial]

        # Limites fisicamente plausíveis
        bounds = [(1e-4, 1e-1), (1e-10, 1e-6)]

        print(f"Valores iniciais: alpha={alpha_inicial:.2e}, beta={beta_inicial:.2e}")

        # Executar otimização
        resultado = minimize(
            self.funcao_custo,
            x0,
            method=metodo,
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )

        alpha_opt, beta_opt = resultado.x

        print(f"✅ Otimização concluída!")
        print(f"   Alpha: {alpha_inicial:.2e} -> {alpha_opt:.2e}")
        print(f"   Beta:  {beta_inicial:.2e} -> {beta_opt:.2e}")
        print(f"   Erro final (RMSE): {resultado.fun:.2f} nm")

        # Gerar simulação final com parâmetros otimizados
        params_otimizados = self.params_base.copy()
        params_otimizados['alpha'] = alpha_opt
        params_otimizados['beta'] = beta_opt

        simulacao_otimizada = run_simulation(params_otimizados)

        # Analisar simulação otimizada
        from graficos import analisar_cores_para_comprimento_onda
        analise_otimizada = analisar_cores_para_comprimento_onda(
            simulacao_otimizada['colors_rgb'],
            simulacao_otimizada['x_cm']
        )

        return {
            'params_otimizados': params_otimizados,
            'resultado_otimizacao': resultado,
            'simulacao_otimizada': simulacao_otimizada,
            'analise_otimizada': analise_otimizada,
            'erro_final': resultado.fun
        }

    def plotar_comparacao(self, resultado_otimizacao):
        """
        Plota comparação entre dados reais, simulação inicial e simulação otimizada
        """
        analise_otimizada = resultado_otimizacao['analise_otimizada']

        plt.figure(figsize=(12, 8))

        # Plot principal: comprimentos de onda
        plt.subplot(2, 1, 1)
        plt.plot(self.dados_reais['posicoes_cm'], self.dados_reais['comprimentos_onda_nm'],
                 'ko-', linewidth=2, markersize=3, label='Dados Reais (Foto)', alpha=0.8)
        plt.plot(self.dados_sim_iniciais['posicoes_cm'], self.dados_sim_iniciais['comprimentos_onda_nm'],
                 'r--', linewidth=2, label='Simulação Inicial', alpha=0.7)
        plt.plot(analise_otimizada['posicoes_cm'], analise_otimizada['comprimentos_onda_nm'],
                 'b-', linewidth=2, label='Simulação Otimizada', alpha=0.8)

        plt.xlabel('Posição (cm)')
        plt.ylabel('Comprimento de Onda (nm)')
        plt.title('Comparação: Otimização de Parâmetros α e β')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot das espessuras
        plt.subplot(2, 1, 2)
        params_ini = self.params_base
        params_opt = resultado_otimizacao['params_otimizados']

        # Simulação inicial
        sim_ini = run_simulation(params_ini)
        plt.plot(sim_ini['x_cm'], sim_ini['thickness_nm'],
                 'r--', linewidth=2, label=f'Inicial (α={params_ini["alpha"]:.2e})', alpha=0.7)

        # Simulação otimizada
        sim_opt = resultado_otimizacao['simulacao_otimizada']
        plt.plot(sim_opt['x_cm'], sim_opt['thickness_nm'],
                 'b-', linewidth=2, label=f'Otimizado (α={params_opt["alpha"]:.2e})', alpha=0.8)

        plt.xlabel('Posição (cm)')
        plt.ylabel('Espessura (nm)')
        plt.title('Perfil de Espessura do Filme - Antes e Depois da Otimização')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()