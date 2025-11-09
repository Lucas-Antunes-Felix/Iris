
from simulacao import run_simulation
from graficos import (
    plot_color_map_only,
    plot_color_map_with_thickness,
    analisar_cores_para_comprimento_onda
)
from analisador_foto import main_analise_fotos, carregar_dados_existentes
from otimizador_dados import executar_otimizacao_inteligente, analisar_erros_manual
from incerteza import executar_analise_incertezas
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# CONFIGURAÇÕES SIMPLIFICADAS
# =============================================================================

PARAMS_PADRAO = {
    # Parâmetros físicos
    'h0': 16692.5e-9,
    'alpha': 0.064662,
    'beta': 1.00e-08,
    'n_film': 1.375,

    # ⚠️ CORREÇÃO: Adicionar tempo inicial controlável
    't_initial': 90,  # TEMPO INICIAL DA SIMULAÇÃO (em segundos)

    # Configurações de análise
    'num_steps': 1000,
    'tempos_analise': [60, 70, 80, 90],  # Para análise de fotos
    'regiao_comparacao_cm': 2.0,

    # Restrições físicas
    'h0_max_observado': 20000e-9,
    'alpha_min_observado': 0.05,
    'alpha_max_observado': 0.07,
    'beta_ideal_observado': 1e-8,
}

# CONTROLE SIMPLIFICADO - ESCOLHA O QUE EXECUTAR
EXECUTAR = {
    'SIMULACAO': True,  # Gera APENAS mapa de cores da simulação (Gráfico 1)
    'ANALISAR_FOTOS': True,  # Analisa fotos reais e salva dados + Gráfico 2
    'TESTAR_PARAMETROS': True,  # Testa parâmetros manuais + Gráficos 3 e 4
    'OTIMIZAR': False,  # Otimização automática completa + Gráficos 3 e 4 otimizados
}

# ⚠️ CORREÇÃO: VARIÁVEL GLOBAL PARA CONTROLAR O TEMPO
TEMPO_SIMULACAO = 60  # Mude este valor para alterar o tempo inicial (60, 70, 80, 90, etc)


# =============================================================================
# FUNÇÕES SIMPLIFICADAS - CORRIGIDAS
# =============================================================================

def executar_simulacao():
    """Executa APENAS a simulação física e gera Gráfico 1"""
    print("🎮 EXECUTANDO SIMULAÇÃO - GERANDO GRÁFICO 1")
    print("=" * 40)
    print(f"⚙️  Tempo inicial: {TEMPO_SIMULACAO}s")
    print("SAÍDA: Mapa de cores da interferência (Gráfico 1)")
    print("-" * 40)

    # ⚠️ CORREÇÃO: Usar TEMPO_SIMULACAO global
    params_simulacao = {
        'h0': PARAMS_PADRAO['h0'],
        'alpha': PARAMS_PADRAO['alpha'],
        'beta': PARAMS_PADRAO['beta'],
        'n_film': PARAMS_PADRAO['n_film'],
        'num_steps': PARAMS_PADRAO['num_steps'],
        't_initial': TEMPO_SIMULACAO  # ⚠️ Usar variável controlável
    }

    # Executar simulação
    resultados_sim = run_simulation(params_simulacao)

    # Gráfico 1: Mapa de cores APENAS
    print("🎨 Gerando Gráfico 1: Mapa de cores da interferência...")
    fig1 = plot_color_map_only(resultados_sim['colors_rgb'], resultados_sim['x_cm'],
                               title=f"Mapa de Cor - t = {TEMPO_SIMULACAO}s")

    # Opcional: Gráfico com espessura
    print("📊 Gerando Gráfico 1b: Mapa com curva de espessura...")
    fig2 = plot_color_map_with_thickness(
        resultados_sim['thickness_nm'],
        resultados_sim['colors_rgb'],
        resultados_sim['x_cm'],
        title=f"Padrão de Cores e Espessura - t = {TEMPO_SIMULACAO}s"
    )

    print("✅ Simulação concluída - Gráfico 1 gerado")
    print(f"   • Tempo simulado: {TEMPO_SIMULACAO}s")
    print(f"   • Espessura inicial: {resultados_sim['thickness_nm'][0]:.1f} nm")
    print(f"   • Espessura final: {resultados_sim['thickness_nm'][-1]:.1f} nm")

    return [fig1, fig2]


def executar_analise_fotos():
    """Analisa as fotos reais automaticamente e gera Gráfico 2"""
    print("📸 ANALISANDO FOTOS REAIS - GERANDO GRÁFICO 2")
    print("=" * 40)
    print(f"Tempos: {PARAMS_PADRAO['tempos_analise']}s")
    print("SAÍDA: Espectros reais vs simulados (Gráfico 2)")
    print("-" * 40)

    # ⚠️ CORREÇÃO: Garantir que tempos_analise esteja correto
    params_analise = PARAMS_PADRAO.copy()
    params_analise['tempos_analise'] = [60, 70, 80, 90]  # Tempos fixos para análise

    resultados = main_analise_fotos(
        params_simulacao=params_analise,
        salvar_dados=True,
        plotar_graficos=True  # Gera Gráfico 2
    )

    if resultados:
        print(f"✅ {len(resultados)} fotos analisadas")
        print("   • Gráfico 2: Espectros reais vs simulados gerado")
        print("   • Dados salvos para análise de incertezas")

        # Resumo rápido
        print("\n📊 RESUMO DOS ESPECTROS:")
        for tempo, dados in resultados.items():
            h_approx = (dados['lambda_medio'] * 1e-9 * 1) / (2 * 1.375) * 1e9  # em nm
            print(f"   t={tempo}s → λ={dados['lambda_medio']:.1f}nm → h≈{h_approx:.1f}nm")

    return resultados


def testar_parametros_manuais():
    """Testa parâmetros específicos com análise completa - Gera Gráficos 3 e 4"""
    print("🧪 TESTANDO PARÂMETROS MANUAIS - GERANDO GRÁFICOS 3 e 4")
    print("=" * 40)
    print(f"h0: {PARAMS_PADRAO['h0'] * 1e9:.1f} nm")
    print(f"α:  {PARAMS_PADRAO['alpha']:.3f}")
    print(f"β:  {PARAMS_PADRAO['beta']:.2e}")
    print("SAÍDA: Gráfico 3 (faixas + pontos) + Gráfico 4 (barras de erro)")
    print("-" * 40)

    # Verificar se dados existem, se não, analisar fotos
    arquivos_necessarios = [f"dados_reais_t{tempo}.json" for tempo in PARAMS_PADRAO['tempos_analise']]
    arquivos_faltantes = [f for f in arquivos_necessarios if not os.path.exists(f)]

    if arquivos_faltantes:
        print(f"📸 {len(arquivos_faltantes)} arquivos faltando, analisando fotos primeiro...")
        executar_analise_fotos()

    # Executar análise de erros (gera Gráficos 3 e 4)
    resultados = analisar_erros_manual(PARAMS_PADRAO, PARAMS_PADRAO)

    if resultados:
        print("✅ Análise manual concluída")
        print("   • Gráfico 3: Espectros com faixas de cor e pontos médios")
        print("   • Gráfico 4: Comparação de pontos médios com barras de erro")

    return resultados


def executar_otimizacao_completa():
    """Executa otimização completa - Gera Gráficos 3 e 4 otimizados"""
    print("🚀 INICIANDO OTIMIZAÇÃO COMPLETA")
    print("=" * 40)
    print("SAÍDA: Parâmetros otimizados + Gráficos 3 e 4 otimizados")
    print("-" * 40)

    # Carregar dados reais
    dados_reais = carregar_dados_existentes()
    if not dados_reais:
        print("❌ Dados reais não encontrados. Executando análise de fotos...")
        dados_reais = executar_analise_fotos()
        if not dados_reais:
            return None

    # Executar otimização
    resultado = executar_otimizacao_inteligente(dados_reais, PARAMS_PADRAO)

    if resultado:
        params_opt = resultado['params_otimizados']
        print(f"\n🎯 PARÂMETROS OTIMIZADOS:")
        print(f"   h₀: {params_opt['h0'] * 1e9:.1f} nm")
        print(f"   α:  {params_opt['alpha']:.6f}")
        print(f"   β:  {params_opt['beta']:.2e}")

        # Salvar parâmetros
        with open('parametros_otimizados.txt', 'w') as f:
            f.write(f"h0: {params_opt['h0'] * 1e9:.1f} nm\n")
            f.write(f"alpha: {params_opt['alpha']:.6f}\n")
            f.write(f"beta: {params_opt['beta']:.2e}\n")

        print("💾 Parâmetros salvos em 'parametros_otimizados.txt'")

    return resultado


# =============================================================================
# FUNÇÕES PARA CONTROLE DE TEMPO
# =============================================================================

def configurar_tempo_simulacao(novo_tempo):
    """
    ⚠️ CORREÇÃO: Função para mudar o tempo da simulação facilmente
    """
    global TEMPO_SIMULACAO
    TEMPO_SIMULACAO = novo_tempo
    print(f"⏰ Tempo da simulação configurado para: {TEMPO_SIMULACAO}s")


def simular_tempo_especifico(tempo):
    """
    ⚠️ CORREÇÃO: Função para simular um tempo específico rapidamente
    """
    configurar_tempo_simulacao(tempo)
    return executar_simulacao()


# =============================================================================
# EXECUÇÃO PRINCIPAL SIMPLIFICADA
# =============================================================================

def main():
    """Execução simplificada e intuitiva"""
    print("🔬 SISTEMA DE ANÁLISE - VERSÃO SIMPLIFICADA")
    print("=" * 50)
    print("CONTROLE ATIVADO:")
    for funcao, ativo in EXECUTAR.items():
        status = "✅" if ativo else "❌"
        print(f"   {status} {funcao}")
    print(f"⏰ Tempo da simulação: {TEMPO_SIMULACAO}s")
    print("=" * 50)

    figuras = []

    try:
        # 1. SIMULAÇÃO (APENAS Gráfico 1) - ⚠️ USA TEMPO_SIMULACAO
        if EXECUTAR['SIMULACAO']:
            figuras += executar_simulacao()

        # 2. ANÁLISE DE FOTOS (Gráfico 2) - ⚠️ USA TEMPOS FIXOS [60,70,80,90]
        if EXECUTAR['ANALISAR_FOTOS']:
            executar_analise_fotos()

        # 3. TESTE DE PARÂMETROS (Gráficos 3 e 4) - ⚠️ USA TEMPOS FIXOS
        if EXECUTAR['TESTAR_PARAMETROS']:
            testar_parametros_manuais()

        # 4. OTIMIZAÇÃO (Gráficos 3 e 4 otimizados) - ⚠️ USA TEMPOS FIXOS
        if EXECUTAR['OTIMIZAR']:
            executar_otimizacao_completa()

        # RESUMO FINAL
        print(f"\n✅ EXECUÇÃO CONCLUÍDA!")
        print("=" * 30)

        if EXECUTAR['SIMULACAO']:
            print(f"🎨 Gráfico 1: Mapas de cor (t={TEMPO_SIMULACAO}s)")

        if EXECUTAR['ANALISAR_FOTOS']:
            print("📊 Gráfico 2: Espectros reais vs simulados")
            print("📁 Dados salvos: dados_reais_t[60,70,80,90].json")

        if EXECUTAR['TESTAR_PARAMETROS']:
            print("📈 Gráfico 3: Espectros com faixas e pontos médios")
            print("📈 Gráfico 4: Pontos médios com barras de erro")

        if EXECUTAR['OTIMIZAR']:
            print("⚙️  Parâmetros otimizados salvos")
            print("📈 Gráficos 3 e 4 otimizados gerados")

        # Manter gráficos abertos apenas para simulação
        if figuras and EXECUTAR['SIMULACAO']:
            print(f"\n🖼️  Gráficos do tempo {TEMPO_SIMULACAO}s gerados")
            print("   Feche as janelas para finalizar...")
            plt.show(block=True)
        elif any([EXECUTAR['ANALISAR_FOTOS'], EXECUTAR['TESTAR_PARAMETROS'], EXECUTAR['OTIMIZAR']]):
            print("\n📊 Gráficos gerados e exibidos durante o processo")

    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MODOS DE OPERAÇÃO RÁPIDOS - CORRIGIDOS
# =============================================================================

def modo_apenas_simulacao(tempo=60):
    """Apenas gera Gráfico 1 (mapas de cor) para tempo específico"""
    global EXECUTAR, TEMPO_SIMULACAO
    EXECUTAR = {
        'SIMULACAO': True,
        'ANALISAR_FOTOS': False,
        'TESTAR_PARAMETROS': False,
        'OTIMIZAR': False,
    }
    TEMPO_SIMULACAO = tempo
    print(f"🎮 MODO: Apenas Simulação (Gráfico 1) - t = {tempo}s")
    main()


def modo_apenas_analise_fotos():
    """Apenas analisa fotos e gera Gráfico 2"""
    global EXECUTAR
    EXECUTAR = {
        'SIMULACAO': False,
        'ANALISAR_FOTOS': True,
        'TESTAR_PARAMETROS': False,
        'OTIMIZAR': False,
    }
    print("📸 MODO: Apenas Análise de Fotos (Gráfico 2)")
    main()


def modo_apenas_teste_parametros():
    """Apenas testa parâmetros e gera Gráficos 3 e 4"""
    global EXECUTAR
    EXECUTAR = {
        'SIMULACAO': False,
        'ANALISAR_FOTOS': False,
        'TESTAR_PARAMETROS': True,
        'OTIMIZAR': False,
    }
    print("🧪 MODO: Apenas Teste de Parâmetros (Gráficos 3 e 4)")
    main()


def modo_apenas_otimizacao():
    """Apenas otimização completa"""
    global EXECUTAR
    EXECUTAR = {
        'SIMULACAO': False,
        'ANALISAR_FOTOS': False,
        'TESTAR_PARAMETROS': False,
        'OTIMIZAR': True,
    }
    print("🚀 MODO: Apenas Otimização (Gráficos 3 e 4 otimizados)")
    main()


def modo_fluxo_completo():
    """Executa fluxo completo: Fotos → Teste → Otimização"""
    global EXECUTAR
    EXECUTAR = {
        'SIMULACAO': False,
        'ANALISAR_FOTOS': True,
        'TESTAR_PARAMETROS': True,
        'OTIMIZAR': True,
    }
    print("🔬 MODO: Fluxo Completo (Gráficos 2, 3, 4 + Otimização)")
    main()


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    print("🔧 CONTROLE DE TEMPO DA SIMULAÇÃO")
    print("=" * 60)
    print("⚠️  PARA MUDAR O TEMPO DA SIMULAÇÃO:")
    print("   1. Edite a variável TEMPO_SIMULACAO (linha ~45)")
    print("   2. Ou use: configurar_tempo_simulacao(novo_tempo)")
    print("   3. Ou use: modo_apenas_simulacao(tempo_desejado)")
    print("\n⏰ Tempos sugeridos: 60, 70, 80, 90, 100, 120")
    print(f"⏰ Tempo atual: {TEMPO_SIMULACAO}s")
    print("=" * 60)

    print("\n🔧 SELECIONE O MODO DE OPERAÇÃO:")
    print("1. modo_apenas_simulacao(60) - Gráfico 1 (t=60s)")
    print("2. modo_apenas_simulacao(80) - Gráfico 1 (t=80s)")
    print("3. modo_apenas_simulacao(120) - Gráfico 1 (t=120s)")
    print("4. modo_apenas_analise_fotos() - Gráfico 2")
    print("5. modo_apenas_teste_parametros() - Gráficos 3 e 4")
    print("6. modo_apenas_otimizacao() - Gráficos 3 e 4 otimizados")
    print("7. main() - Usar configuração EXECUTAR atual")
    print("\n💡 Dica: Mude TEMPO_SIMULACAO para controlar o tempo")
    print("=" * 60)

    # Exemplo rápido: simular t=80s
    # modo_apenas_simulacao(80)

    # Ou usar configuração atual
    main()