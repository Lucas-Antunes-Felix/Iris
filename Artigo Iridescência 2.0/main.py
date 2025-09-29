# =============================================================================
# MAIN.PY - CONTROLE PRINCIPAL DO SISTEMA
# =============================================================================

from simulacao import run_simulation
from graficos import (
    plot_color_map_only,
    plot_color_map_with_thickness,
    analisar_cores_para_comprimento_onda,
    plotar_comprimento_onda_vs_posicao,
    aguardar_fechamento  # ← Mudança no nome da função
)
from analisador_foto import AnalisadorFotoCalibrado

# =============================================================================
# CONFIGURAÇÕES - AJUSTE AQUI
# =============================================================================

# Parâmetros da simulação física
PARAMS_SIMULACAO = {
    'h0': 1450e-9,  # Espessura inicial (m)
    'alpha': 1.1e-2,  # Coeficiente de decaimento
    'beta': 0.9e-8,  # Taxa de Evaporação
    'n_film': 1.3,  # Índice de refração do filme
    'num_steps': 1000,  # Número de pontos
    't_initial': 65  # Tempo Final
}

# Flags de controle
FLAGS = {
    # Simulação computacional
    'MAPA_COR': True,
    'MAPA_COR_ESPESSURA': False,
    'COMPRIMENTOS_ONDA': True,

    # Análise de foto real
    'ANALISE_FOTO': False,

    # Configurações da foto
    'CAMINHO_FOTO': "/Users/macbook/Desktop/padraoreal.png",
    'ORIENTACAO_FOTO': 'horizontal'
}


# =============================================================================
# EXECUÇÃO - NÃO MODIFICAR
# =============================================================================

def main():
    """Executa as análises conforme configuração"""
    print("SISTEMA DE ANÁLISE DE INTERFERÊNCIA ÓPTICA")

    # Lista para armazenar referências das figuras
    figuras = []

    # Executar simulação
    if any([FLAGS['MAPA_COR'], FLAGS['MAPA_COR_ESPESSURA'], FLAGS['COMPRIMENTOS_ONDA']]):
        resultados_sim = run_simulation(PARAMS_SIMULACAO)

        if FLAGS['MAPA_COR']:
            fig = plot_color_map_only(resultados_sim['colors_rgb'], resultados_sim['x_cm'])
            figuras.append(fig)

        if FLAGS['MAPA_COR_ESPESSURA']:
            fig = plot_color_map_with_thickness(
                resultados_sim['thickness_nm'],
                resultados_sim['colors_rgb'],
                resultados_sim['x_cm']
            )
            figuras.append(fig)

        if FLAGS['COMPRIMENTOS_ONDA']:
            analise = analisar_cores_para_comprimento_onda(
                resultados_sim['colors_rgb'],
                resultados_sim['x_cm']
            )
            fig = plotar_comprimento_onda_vs_posicao(analise)
            figuras.append(fig)

    # Analisar foto real
    if FLAGS['ANALISE_FOTO']:
        analisador = AnalisadorFotoCalibrado()
        resultados_sim = run_simulation(PARAMS_SIMULACAO)

        analisador.calibrar_com_simulacao(
            resultados_sim,
            FLAGS['CAMINHO_FOTO'],
            FLAGS['ORIENTACAO_FOTO']
        )

        resultados_foto = analisador.analisar_comprimentos_onda_calibrado()
        # O plotar_comparacao do analisador_foto também precisa ser ajustado
        # Por enquanto, vamos manter o comportamento atual

    print(f"\nTodas as análises concluídas! {len(figuras)} gráficos gerados.")
    print("   Feche todas as janelas para encerrar o programa.")

    # Manter o programa rodando até que todas as figuras sejam fechadas
    aguardar_fechamento()

if __name__ == "__main__":
    main()