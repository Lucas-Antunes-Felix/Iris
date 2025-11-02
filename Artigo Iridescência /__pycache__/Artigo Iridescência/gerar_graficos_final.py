# gerar_graficos_final.py
import numpy as np
import matplotlib.pyplot as plt
from simulacao import run_simulation
from graficos import analisar_cores_para_comprimento_onda
from analisador_foto import main_analise_fotos


def plotar_espectros_comparacao_final():
    """
    Gera gráficos finais com os parâmetros específicos do resultado
    """
    print("📊 GERANDO GRÁFICOS FINAIS...")

    # Parâmetros FIXOS do seu resultado
    params_inicial = {
        'h0': 1600e-9,  # 20000.0 nm
        'alpha': 0.06,  # 0.005
        'beta': 1.0e-08  # 5.00e-08
    }

    params_otimizado = {
        'h0': 16013.7e-9,  # 16013.7 nm
        'alpha': 0.060,  # 0.060
        'beta': 1.02e-08  # 1.02e-08
    }

    # Obter dados experimentais
    print("📷 Obtendo dados experimentais...")
    dados_experimentais = main_analise_fotos()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    tempos = [60, 70, 80, 90]

    for idx, tempo in enumerate(tempos):
        ax = axes[idx]

        # Dados reais
        if tempo in dados_experimentais:
            dados_reais = dados_experimentais[tempo]

            # Simulação INICIAL
            try:
                print(f"🔄 Simulando t={tempo}s (INICIAL)...")
                simulacao_inicial = run_simulation({
                    'h0': params_inicial['h0'],
                    'alpha': params_inicial['alpha'],
                    'beta': params_inicial['beta'],
                    'n_film': 1.375,
                    'num_steps': 1000,
                    't_initial': tempo
                })
                analise_inicial = analisar_cores_para_comprimento_onda(
                    simulacao_inicial['colors_rgb'],
                    simulacao_inicial['x_cm']
                )
            except Exception as e:
                print(f"❌ Erro simulação inicial t={tempo}s: {e}")
                analise_inicial = None

            # Simulação OTIMIZADA
            try:
                print(f"🔄 Simulando t={tempo}s (OTIMIZADO)...")
                simulacao_opt = run_simulation({
                    'h0': params_otimizado['h0'],
                    'alpha': params_otimizado['alpha'],
                    'beta': params_otimizado['beta'],
                    'n_film': 1.375,
                    'num_steps': 1000,
                    't_initial': tempo
                })
                analise_opt = analisar_cores_para_comprimento_onda(
                    simulacao_opt['colors_rgb'],
                    simulacao_opt['x_cm']
                )
            except Exception as e:
                print(f"❌ Erro simulação otimizada t={tempo}s: {e}")
                analise_opt = None

            # Plot das três curvas
            # Dados REAIS
            ax.plot(dados_reais['dados_completos']['posicoes_cm'],
                    dados_reais['dados_completos']['comprimentos_onda_nm'],
                    'ko-', linewidth=2, markersize=3, label='Experimental', alpha=0.8)

            # Simulação OTIMIZADA
            if analise_opt is not None:
                ax.plot(analise_opt['posicoes_cm'], analise_opt['comprimentos_onda_nm'],
                        'b-', linewidth=2, label='Otimizado', alpha=0.8)

            # Simulação INICIAL
            if analise_inicial is not None:
                ax.plot(analise_inicial['posicoes_cm'], analise_inicial['comprimentos_onda_nm'],
                        'r--', linewidth=2, label='Inicial', alpha=0.7)

            ax.set_xlabel('Posição (cm)')
            ax.set_ylabel('Comprimento de Onda (nm)')
            ax.set_title(f't = {tempo}s')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 4)
            ax.set_ylim(400, 700)
        else:
            print(f"⚠️  Dados experimentais para t={tempo}s não encontrados")

    plt.suptitle(
        'Comparação: Inicial (h₀=20000nm, α=0.005, β=5e-08) vs Otimizado (h₀=16013.7nm, α=0.060, β=1.02e-08)',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    print("✅ Gráficos gerados com sucesso!")
    print("🖼️  Fechando a janela do gráfico para finalizar...")

    # Forçar a exibição e bloquear até fechar
    plt.show(block=True)

    print("🎯 Programa finalizado!")


if __name__ == "__main__":
    plotar_espectros_comparacao_final()