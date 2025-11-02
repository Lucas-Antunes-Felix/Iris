# =============================================================================
# MAIN.PY - CONTROLE PRINCIPAL DO SISTEMA (VERSÃO INTELIGENTE)
# =============================================================================

from simulacao import run_simulation
from graficos import (
    plot_color_map_only,
    plot_color_map_with_thickness,
    analisar_cores_para_comprimento_onda,
    plotar_comprimento_onda_vs_posicao
)
from analisador_foto import main_analise_fotos
from otimizador_dados import executar_otimizacao_inteligente
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURAÇÕES CENTRALIZADAS - AJUSTE TODOS OS PARÂMETROS AQUI
# =============================================================================

# PARÂMETROS GLOBAIS DA SIMULAÇÃO E OTIMIZAÇÃO
PARAMS_SIMULACAO = {
    # Parâmetros físicos da simulação
    'h0': 16000.0e-9,
    'alpha': 0.06,
    'beta': 1.0e-08,
    'n_film': 1.375,
    'num_steps': 1000,
    't_initial': 90,

    # NOVAS RESTRIÇÕES BASEADAS NA ANÁLISE OBSERVACIONAL
    'h0_max_observado': 20000e-9,  # 20000 nm - limite físico máximo
    'alpha_min_observado': 0.05,  # Mínimo baseado em t=120s
    'alpha_max_observado': 0.07,  # Máximo baseado em t=60s
    'beta_ideal_observado': 1e-8,  # Valor ideal baseado na observação
    'tolerancia_beta_observado': 0.3,  # ±30% de tolerância para beta

    # Parâmetros de otimização
    'max_iter': 50,
    'h0_inicial': 20000e-9,
    'alpha_inicial': 0.005,
    'beta_inicial': 5e-08,

    # Configurações do black film
    'limite_variacao_black_film': 1.5,
    'fator_adiantamento_segundos': 5,
    'num_pontos_black_film': 3,

    # Regiões de análise
    'regiao_comparacao_cm': 2.0,
    'posicoes_comparacao_pontos': 500,

    # Tempos para análise
    'tempos_analise': [60, 70, 80, 90],
    'tempo_deteccao_black_film': 60,

    # Configurações do algoritmo
    'max_iter_deteccao': 50,
    'fator_reducao_h0': 0.95,
    'fator_aumento_alpha': 1.03,
    'fator_aumento_beta': 1.02,

    # Fatores de ajuste para black film mais cedo
    'fator_reducao_h0_adiantamento': 0.85,
    'fator_aumento_alpha_adiantamento': 1.15,
    'fator_aumento_beta_adiantamento': 1.15,

    # Limites físicos
    'h0_min': 100e-9,
    'alpha_max': 0.5,
    'beta_max': 1e-06,
    'alpha_min': 1e-6,
    'beta_min': 1e-12
}

# Flags de controle - ATIVE/DESATIVE AS ANÁLISES AQUI
FLAGS = {
    # Simulação computacional
    'MAPA_COR': False,
    'MAPA_COR_ESPESSURA': False,
    'COMPRIMENTOS_ONDA': False,

    # Análise de foto real
    'ANALISE_FOTO': True,  # ← ATIVAR para analisar fotos

    # Otimização de parâmetros (AGORA INTELIGENTE)
    'OTIMIZAR_ALPHA_BETA': True,  # ← ATIVAR para otimização inteligente

    # Configurações da foto
    'CAMINHO_FOTO': "/Users/macbook/Desktop/padraoreal.png",
    'ORIENTACAO_FOTO': 'horizontal'
}


# =============================================================================
# EXECUÇÃO - MODIFICADO PARA ESTRATÉGIA INTELIGENTE
# =============================================================================

def main():
    """Executa as análises conforme configuração (VERSÃO INTELIGENTE)"""
    print("🔬 SISTEMA DE ANÁLISE DE INTERFERÊNCIA ÓPTICA - ESTRATÉGIA INTELIGENTE")
    print("======================================================================")
    print("🎯 ESTRATÉGIA: Detectar picos automaticamente + Otimizar apenas na faixa com cores")
    print("=" * 80)

    # Lista para armazenar figuras
    figuras = []

    # Executar simulação (se necessário para alguma análise)
    resultados_sim = None
    if any([FLAGS['MAPA_COR'], FLAGS['MAPA_COR_ESPESSURA'], FLAGS['COMPRIMENTOS_ONDA'], FLAGS['OTIMIZAR_ALPHA_BETA']]):
        print("\n🔄 Executando simulação física...")
        # Usar apenas os parâmetros de simulação físicos
        params_fisicos = {k: PARAMS_SIMULACAO[k] for k in ['h0', 'alpha', 'beta', 'n_film', 'num_steps', 't_initial']}
        resultados_sim = run_simulation(params_fisicos)
        print("✅ Simulação concluída")

    # Análises de simulação
    if FLAGS['MAPA_COR']:
        print("\n📊 Gerando mapa de cores da simulação...")
        fig = plot_color_map_only(resultados_sim['colors_rgb'], resultados_sim['x_cm'])
        figuras.append(fig)

    if FLAGS['MAPA_COR_ESPESSURA']:
        print("\n📊 Gerando mapa de cores com espessura...")
        fig = plot_color_map_with_thickness(
            resultados_sim['thickness_nm'],
            resultados_sim['colors_rgb'],
            resultados_sim['x_cm']
        )
        figuras.append(fig)

    if FLAGS['COMPRIMENTOS_ONDA']:
        print("\n🌈 Analisando comprimentos de onda da simulação...")
        analise_sim = analisar_cores_para_comprimento_onda(
            resultados_sim['colors_rgb'],
            resultados_sim['x_cm']
        )
        fig = plotar_comprimento_onda_vs_posicao(analise_sim)
        figuras.append(fig)

    # ANÁLISE DAS 4 FOTOS TEMPORAIS (ÚLTIMOS PADRÕES COLORIDOS)
    resultados_fotos = None
    if FLAGS['ANALISE_FOTO']:
        print("\n🎯 ANALISANDO OS PADRÕES COLORIDOS (60, 70, 80, 90s)")
        print("   Estratégia: Detectar automaticamente a faixa com picos")
        print("=" * 60)

        # Executar análise das 4 fotos
        resultados_fotos = main_analise_fotos()

        print("\n📊 ESPECTROS DOS PADRÕES COLORIDOS:")
        print("=" * 50)
        for tempo, dados in resultados_fotos.items():
            # Calcular espessura aproximada
            h_approx = (dados['lambda_medio'] * 1e-9 * 1) / (2 * 1.375) * 1e9
            print(f"t = {tempo}s → λ = {dados['lambda_medio']:.1f} nm → h ≈ {h_approx:.1f} nm")

        print("\n📊 AGUARDANDO FECHAMENTO DO GRÁFICO...")
        print("   Feche a janela do gráfico para continuar...")
        plt.show(block=True)

    # 🎯 OTIMIZAÇÃO INTELIGENTE - DETECÇÃO AUTOMÁTICA DE PICOS
    if FLAGS['OTIMIZAR_ALPHA_BETA'] and resultados_fotos:
        print("\n🎯 INICIANDO OTIMIZAÇÃO INTELIGENTE")
        print("   ESTRATÉGIA: Detectar picos automaticamente + Focar apenas na faixa com cores")
        print("=" * 80)
        print("📋 MÉTODO:")
        print("   1. Começar com tempos altos (black film completo)")
        print("   2. Diminuir tempo até detectar PRIMEIROS PICOS")
        print("   3. Otimizar APENAS na faixa temporal com picos")
        print("   4. Ignorar tempos com apenas bege constante")
        print("-" * 80)

        print(f"⚙️  PARÂMETROS DE OTIMIZAÇÃO:")
        print(f"   h0_inicial: {PARAMS_SIMULACAO['h0_inicial'] * 1e9:.1f} nm")
        print(f"   alpha_inicial: {PARAMS_SIMULACAO['alpha_inicial']:.3f}")
        print(f"   beta_inicial: {PARAMS_SIMULACAO['beta_inicial']:.2e}")
        print(f"   max_iter: {PARAMS_SIMULACAO['max_iter']}")
        print(f"   Tempos experimentais: {PARAMS_SIMULACAO['tempos_analise']}s")
        print("-" * 80)

        try:
            # Executar otimização INTELIGENTE com TODOS os parâmetros
            resultado_inteligente = executar_otimizacao_inteligente(
                dados_temporais=resultados_fotos,
                params_simulacao=PARAMS_SIMULACAO  # ← Passar TODOS os parâmetros
            )

            params_opt = resultado_inteligente['params_otimizados']
            print(f"\n🎯 PARÂMETROS OTIMIZADOS (ESTRATÉGIA INTELIGENTE):")
            print(f"   h₀ = {params_opt['h0'] * 1e9:.1f} nm")
            print(f"   α  = {params_opt['alpha']:.6f} s⁻¹")
            print(f"   β  = {params_opt['beta']:.2e} m/s")
            print(f"   Erro final: {resultado_inteligente['erro_final']:.2f}")

            # Verificar restrições
            if resultado_inteligente['restricoes_respeitadas']:
                print("   ✅ Todas as restrições foram respeitadas")
            else:
                print("   ⚠️  Algumas restrições não foram respeitadas")

            # Salvar parâmetros otimizados
            with open('parametros_otimizados_inteligente.txt', 'w') as f:
                f.write("PARÂMETROS OTIMIZADOS - ESTRATÉGIA INTELIGENTE\n")
                f.write("==============================================\n")
                f.write(f"h0:     {params_opt['h0'] * 1e9:.1f} nm\n")
                f.write(f"alpha:  {params_opt['alpha']:.6f} s⁻¹\n")
                f.write(f"beta:   {params_opt['beta']:.2e} m/s\n")
                f.write(f"erro:   {resultado_inteligente['erro_final']:.2f}\n")
                f.write(f"restricoes_respeitadas: {resultado_inteligente['restricoes_respeitadas']}\n")
                f.write(f"\n# Configurações usadas:\n")
                f.write(f"# max_iter: {PARAMS_SIMULACAO['max_iter']}\n")
                f.write(f"# tempos_analise: {PARAMS_SIMULACAO['tempos_analise']}\n")
                f.write(f"# fator_adiantamento: {PARAMS_SIMULACAO['fator_adiantamento_segundos']}s\n")

            print("✅ Parâmetros salvos em 'parametros_otimizados_inteligente.txt'")

        except Exception as e:
            print(f"❌ Erro na otimização inteligente: {e}")
            import traceback
            traceback.print_exc()

    # RESUMO FINAL
    print(f"\n✅ TODAS AS ANÁLISES CONCLUÍDAS!")
    analises_realizadas = []

    if FLAGS['MAPA_COR']:
        analises_realizadas.append("Mapa de Cores")
    if FLAGS['MAPA_COR_ESPESSURA']:
        analises_realizadas.append("Mapa com Espessura")
    if FLAGS['COMPRIMENTOS_ONDA']:
        analises_realizadas.append("Comprimentos de Onda Simulados")
    if FLAGS['ANALISE_FOTO']:
        analises_realizadas.append("Análise dos Padrões Coloridos")
    if FLAGS['OTIMIZAR_ALPHA_BETA'] and resultados_fotos:
        analises_realizadas.append("Otimização Inteligente (Detecção de Picos)")

    print(f"   Análises realizadas: {', '.join(analises_realizadas)}")

    # GARANTIR QUE GRÁFICOS APAREÇAM
    if FLAGS['OTIMIZAR_ALPHA_BETA'] and resultados_fotos:
        print("📊 Gráficos de comparação devem estar abertos...")
        print("   Feche a janela do gráfico para finalizar...")
        plt.show(block=True)
    elif figuras:
        print(f"   {len(figuras)} gráficos gerados")
        plt.show(block=True)
    else:
        print("   Nenhum gráfico para exibir")


if __name__ == "__main__":
    main()