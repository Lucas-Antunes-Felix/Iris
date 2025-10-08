# [file name]: main.py
# =============================================================================
# MAIN.PY - CONTROLE PRINCIPAL DO SISTEMA
# =============================================================================

from simulacao import run_simulation
from graficos import (
    plot_color_map_only,
    plot_color_map_with_thickness,
    analisar_cores_para_comprimento_onda,
    plotar_comprimento_onda_vs_posicao
)
from analisador_foto import AnalisadorFotoCalibrado
from otimizador_dados_existentes import OtimizadorDadosExistentes
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURAÇÕES - AJUSTE AQUI
# =============================================================================

# Parâmetros da simulação física
PARAMS_SIMULACAO = {
    'h0': 1450e-9,  # Espessura inicial (m)
    'alpha': 1.185342e-02,  # Coeficiente de decaimento
    'beta': 1.076522e-08,  # Taxa de evaporação
    'n_film': 1.3,  # Índice de refração do filme
    'num_steps': 1000,  # Número de pontos
    't_initial': 85  # Tempo Final
}

# Flags de controle - ATIVE/DESATIVE AS ANÁLISES AQUI
FLAGS = {
    # Simulação computacional
    'MAPA_COR': False,
    'MAPA_COR_ESPESSURA': True,
    'COMPRIMENTOS_ONDA': True,

    # Análise de foto real
    'ANALISE_FOTO': True,

    # Otimização de parâmetros
    'OTIMIZAR_ALPHA_BETA': False,

    # Configurações da foto
    'CAMINHO_FOTO': "/Users/macbook/Desktop/padraoreal.png",
    'ORIENTACAO_FOTO': 'horizontal'
}


# =============================================================================
# EXECUÇÃO - NÃO MODIFICAR
# =============================================================================

def executar_otimizacao_alpha_beta(resultados_sim, resultados_foto):
    """Executa a otimização dos parâmetros alpha e beta"""
    print("\n🎯 INICIANDO OTIMIZAÇÃO DE α E β...")

    # Analisar simulação inicial para obter dados de comprimento de onda
    from graficos import analisar_cores_para_comprimento_onda
    analise_sim = analisar_cores_para_comprimento_onda(
        resultados_sim['colors_rgb'],
        resultados_sim['x_cm']
    )

    print(f"📊 Dados da simulação: {len(analise_sim['comprimentos_onda_nm'])} pontos")
    print(f"📊 Dados da foto: {len(resultados_foto['comprimentos_onda_nm'])} pontos")

    # Criar e executar otimizador
    otimizador = OtimizadorDadosExistentes(
        dados_reais=resultados_foto,
        dados_simulados_iniciais=analise_sim,
        params_base=PARAMS_SIMULACAO
    )

    resultado_otimizacao = otimizador.otimizar(
        alpha_inicial=PARAMS_SIMULACAO['alpha'],
        beta_inicial=PARAMS_SIMULACAO['beta'],
        metodo='Nelder-Mead',
        max_iter=50
    )

    # Mostrar resultados
    print("\n📈 GERANDO GRÁFICOS DA OTIMIZAÇÃO...")
    otimizador.plotar_comparacao(resultado_otimizacao)

    # Salvar parâmetros otimizados
    params_opt = resultado_otimizacao['params_otimizados']
    with open('parametros_otimizados.txt', 'w') as f:
        f.write("PARÂMETROS OTIMIZADOS - SISTEMA DE INTERFERÊNCIA\n")
        f.write("================================================\n")
        f.write(f"Alpha:  {params_opt['alpha']:.6e} (inicial: {PARAMS_SIMULACAO['alpha']:.6e})\n")
        f.write(f"Beta:   {params_opt['beta']:.6e} (inicial: {PARAMS_SIMULACAO['beta']:.6e})\n")
        f.write(f"h0:     {params_opt['h0']:.6e}\n")
        f.write(f"n_film: {params_opt['n_film']:.3f}\n")
        f.write(f"Erro RMSE final: {resultado_otimizacao['erro_final']:.2f} nm\n")
        f.write(
            f"Melhoria: {((PARAMS_SIMULACAO['alpha'] - params_opt['alpha']) / PARAMS_SIMULACAO['alpha'] * 100):+.1f}% (alpha)\n")
        f.write(
            f"Melhoria: {((PARAMS_SIMULACAO['beta'] - params_opt['beta']) / PARAMS_SIMULACAO['beta'] * 100):+.1f}% (beta)\n")

    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print(f"   Alpha otimizado: {params_opt['alpha']:.6e}")
    print(f"   Beta otimizado:  {params_opt['beta']:.6e}")
    print(f"   Erro final: {resultado_otimizacao['erro_final']:.2f} nm")
    print(f"   Parâmetros salvos em 'parametros_otimizados.txt'")

    return resultado_otimizacao


def main():
    """Executa as análises conforme configuração"""
    print("🔬 SISTEMA DE ANÁLISE DE INTERFERÊNCIA ÓPTICA")
    print("=============================================")

    # Lista para armazenar figuras
    figuras = []

    # Executar simulação (se necessário para alguma análise)
    resultados_sim = None
    if any([FLAGS['MAPA_COR'], FLAGS['MAPA_COR_ESPESSURA'], FLAGS['COMPRIMENTOS_ONDA'], FLAGS['OTIMIZAR_ALPHA_BETA']]):
        print("\n🔄 Executando simulação física...")
        resultados_sim = run_simulation(PARAMS_SIMULACAO)
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

    # Analisar foto real
    resultados_foto = None
    if FLAGS['ANALISE_FOTO'] or FLAGS['OTIMIZAR_ALPHA_BETA']:
        print("\n📷 Iniciando análise de foto...")
        analisador = AnalisadorFotoCalibrado()

        if resultados_sim is None:
            resultados_sim = run_simulation(PARAMS_SIMULACAO)

        print("🔧 Calibrando análise de foto...")
        analisador.calibrar_com_simulacao(
            resultados_sim,
            FLAGS['CAMINHO_FOTO'],
            FLAGS['ORIENTACAO_FOTO']
        )

        print("📊 Analisando comprimentos de onda da foto...")
        resultados_foto = analisador.analisar_comprimentos_onda_calibrado()

        if FLAGS['ANALISE_FOTO']:
            print("📈 Gerando gráficos da análise de foto...")
            analisador.plotar_comparacao(resultados_sim)

    # Otimização de alpha e beta
    if FLAGS['OTIMIZAR_ALPHA_BETA']:
        if resultados_sim is None:
            resultados_sim = run_simulation(PARAMS_SIMULACAO)
        if resultados_foto is None:
            # Se não foi feita análise de foto ainda, fazer agora
            print("\n📷 Preparando análise de foto para otimização...")
            analisador = AnalisadorFotoCalibrado()
            analisador.calibrar_com_simulacao(
                resultados_sim,
                FLAGS['CAMINHO_FOTO'],
                FLAGS['ORIENTACAO_FOTO']
            )
            resultados_foto = analisador.analisar_comprimentos_onda_calibrado()

        # Executar otimização
        resultado_otimizacao = executar_otimizacao_alpha_beta(resultados_sim, resultados_foto)
        figuras.extend(plt.get_fignums())  # Adicionar figuras da otimização

    # Resumo final
    print(f"\n✅ TODAS AS ANÁLISES CONCLUÍDAS!")
    analises_realizadas = []
    if FLAGS['MAPA_COR']: analises_realizadas.append("Mapa de Cores")
    if FLAGS['MAPA_COR_ESPESSURA']: analises_realizadas.append("Mapa com Espessura")
    if FLAGS['COMPRIMENTOS_ONDA']: analises_realizadas.append("Comprimentos de Onda Simulados")
    if FLAGS['ANALISE_FOTO']: analises_realizadas.append("Análise de Foto")
    if FLAGS['OTIMIZAR_ALPHA_BETA']: analises_realizadas.append("Otimização α e β")

    print(f"   Análises realizadas: {', '.join(analises_realizadas)}")
    print(f"   {len(figuras)} gráficos gerados")
    print("   As janelas dos gráficos devem estar abertas.")

    # Manter as figuras abertas
    if figuras:
        plt.show(block=True)
    else:
        print("   Nenhum gráfico para exibir (todas as flags estão False?)")


if __name__ == "__main__":
    main()