from simulacao import run_simulation
from graficos import (
    plot_color_map_only,
    plot_color_map_with_thickness,
    analisar_cores_para_comprimento_onda
)
from analisador_foto import main_analise_fotos, carregar_dados_existentes
from otimizador_dados import executar_otimizacao_inteligente, analisar_erros_manual
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# CONFIGURAÇÕES SIMPLIFICADAS
# =============================================================================

PARAMS_PADRAO = {
    # Parâmetros físicos
    'h0': 12922.9e-9,
    'alpha': 0.056922,
    'beta': 1.00e-09,
     'n_film': 1.375,

    # ⚠️ CORREÇÃO: Adicionar tempo inicial controlável
    't_initial': 80,  # TEMPO INICIAL DA SIMULAÇÃO (em segundos)

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
    'ANALISAR_FOTOS': False,  # Analisa fotos reais e salva dados + Gráfico 2
    'TESTAR_PARAMETROS': False,  # Testa parâmetros manuais + Gráficos 3 e 4
    'OTIMIZAR': False,  # Otimização automática completa + Gráficos 3 e 4 otimizados
}

# ⚠️ CORREÇÃO: VARIÁVEL GLOBAL PARA CONTROLAR O TEMPO
TEMPO_SIMULACAO = 80 # Mude este valor para alterar o tempo inicial (60, 70, 80, 90, etc)


# =============================================================================
# FUNÇÕES SIMPLIFICADAS - CORRIGIDAS
# =============================================================================

def executar_simulacao():
    try:
        params_simulacao = {
            'h0': 12922.9e-9,
            'alpha': 0.056922,
            'beta': 1.00e-09,
            'n_film': 1.375,
            'num_steps': 500,
            't_initial': TEMPO_SIMULACAO,
            'analisar_convolucao': False
        }

        from simulacao import run_simulation_with_convolucion_analysis
        resultados_sim = run_simulation_with_convolucion_analysis(params_simulacao)

        print("🚀 Iniciando simulação...")
        resultados_sim = run_simulation(params_simulacao)

        # 🔥 VERIFICAÇÃO CRÍTICA
        if resultados_sim is None:
            print("❌ ERRO: run_simulation() retornou None!")
            return []

        if not isinstance(resultados_sim, dict):
            print(f"❌ ERRO: run_simulation() retornou {type(resultados_sim)} em vez de dict!")
            return []

        if 'colors_rgb' not in resultados_sim:
            print("❌ ERRO: Chave 'colors_rgb' não encontrada nos resultados!")
            print(f"   Chaves disponíveis: {list(resultados_sim.keys())}")
            return []

        if len(resultados_sim['colors_rgb']) == 0:
            print("❌ ERRO: Array de cores vazio!")
            return []

        print(f"✅ Simulação válida: {len(resultados_sim['colors_rgb'])} cores geradas")

        # 🔥 CORREÇÃO: Usar a função que mostra mapa + curva de espessura
        try:
            # Método 1: Com título personalizado
            fig1 = plot_color_map_with_thickness(
                thickness_array=resultados_sim['thickness_nm'],  # ⚠️ PRIMEIRO PARÂMETRO
                colors_array=resultados_sim['colors_rgb'],  # ⚠️ SEGUNDO PARÂMETRO
                x_cm=resultados_sim['x_cm'],
                title=f"Mapa de Cores com Espessura - Tempo: {TEMPO_SIMULACAO}s"
            )
        except Exception as e:
            print(f"⚠️  Erro na plotagem com espessura: {e}")
            # Método 2: Fallback para mapa simples
            fig1 = plot_color_map_only(
                colors_array=resultados_sim['colors_rgb'],
                x_cm=resultados_sim['x_cm'],
                title=f"Mapa de Cores - Tempo: {TEMPO_SIMULACAO}s"
            )

        return [fig1]

    except Exception as e:
        print(f"❌ Erro em executar_simulacao: {e}")
        import traceback
        traceback.print_exc()
        return []

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
    """Testa parâmetros específicos - VERSÃO COMPATÍVEL"""
    print("🧪 TESTANDO PARÂMETROS - VERSÃO COMPATÍVEL")

    # 1. Carregar dados reais
    dados_reais = carregar_dados_existentes()
    if not dados_reais:
        print("❌ Execute análise de fotos primeiro!")
        return None

    # 2. Gerar dados simulados
    from analisador_foto import AnalisadorEspectros
    analisador_fotos = AnalisadorEspectros()
    dados_simulados = analisador_fotos.carregar_dados_simulados(PARAMS_PADRAO)

    # 3. 🎯 CORREÇÃO DE COMPATIBILIDADE - Usar função que existe
    try:
        # Tentar a função principal primeiro
        from incerteza import executar_analise_incertezas
        resultados = executar_analise_incertezas(
            dados_reais,
            dados_simulados,
            params_simulacao=PARAMS_PADRAO,
            plotar_exemplo=True
        )
    except Exception as e:
        print(f"⚠️  Tentando método alternativo: {e}")
        # Fallback para função compatível
        try:
            from incerteza import executar_analise_incertezas_compativel
            resultados = executar_analise_incertezas_compativel(
                dados_reais,
                dados_simulados
            )
        except Exception as e2:
            print(f"❌ Erro em análise de incertezas: {e2}")
            return None

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
# EXECUÇÃO PRINCIPAL SIMPLIFICADA - CORRIGIDA
# =============================================================================

def main():
    """Execução simplificada e intuitiva - CORRIGIDA"""
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

        # 🔥 CORREÇÃO CRÍTICA: plt.show() DEVE SER CHAMADO APÓS TODOS OS GRÁFICOS
        if figuras or any([EXECUTAR['ANALISAR_FOTOS'], EXECUTAR['TESTAR_PARAMETROS'], EXECUTAR['OTIMIZAR']]):
            print("\n🖼️  MOSTRANDO GRÁFICOS...")
            print("   Feche as janelas para finalizar o programa.")
            plt.show(block=True)  # ⚠️ AGORA ESTÁ NO LUGAR CORRETO!
        else:
            print("\n📊 Nenhum gráfico para exibir.")

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