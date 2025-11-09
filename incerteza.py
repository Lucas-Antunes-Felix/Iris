# incerteza.py - VERSÃO COMPLETA E CORRIGIDA
"""
Módulo para análise de incertezas posicionais - FLUXO CORRETO:
- Entrada: Espectros reais e simulados (Gráfico 2)
- Saída: Gráfico 3 (espectros + faixas + pontos médios)
          Gráfico 4 (comparação pontos médios com barras de erro)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import json
import os


class AnalisadorIncertezas:
    def __init__(self, dados_reais, dados_simulados):
        """
        Analisa incertezas posicionais comparando dados reais vs simulados
        """
        self.dados_reais = dados_reais
        self.dados_simulados = dados_simulados
        self.INCERTEZA_REGUA = 0.05  # cm - incerteza instrumental da régua

        # ⚠️ CORREÇÃO: DICIONÁRIOS SEPARADOS para real vs simulado
        self.config = {
            'limite_black_film': 10,
            'distancia_minima_picos': 15,
            'prominencia_pico': 5,

            # DICIONÁRIO PARA DADOS REAIS (câmera distorce cores)
            'faixas_reais': {
                'violeta': (380, 460),  # ⚠️ Ampliado para câmera
                'azul': (450, 510),  # ⚠️ Ampliado
                'verde': (500, 580),  # ⚠️ Ampliado
                'amarelo': (570, 610),  # ⚠️ Ampliado
                'laranja': (600, 650),  # ⚠️ Ampliado
                'vermelho': (630, 750)  # ⚠️ Ampliado
            },

            # DICIONÁRIO PARA DADOS SIMULADOS (física perfeita)
            'faixas_simulados': {
                'violeta': (380, 440),
                'azul': (440, 490),
                'verde': (490, 570),
                'amarelo': (570, 590),
                'laranja': (590, 620),
                'vermelho': (620, 750)
            },

            'limiar_variacao_black_film': 5.0,
            'min_pontos_faixa': 3,
            'tamanho_janela_black_film': 0.5,
        }

    def classificar_cor(self, comprimento_onda, is_simulado=False):
        """
        CORREÇÃO: Classificação mais robusta para garantir detecção das cores principais
        """
        if is_simulado:
            faixas = self.config['faixas_simulados']
        else:
            faixas = self.config['faixas_reais']

        # Tolerância aumentada para garantir detecção
        tolerancia = 15  # nm de tolerância

        for cor, (min_lam, max_lam) in faixas.items():
            # ⚠️ CORREÇÃO: Ampliar faixas com tolerância
            min_lam_ajustado = min_lam - tolerancia
            max_lam_ajustado = max_lam + tolerancia

            if min_lam_ajustado <= comprimento_onda <= max_lam_ajustado:
                return cor

        # ⚠️ CORREÇÃO: Fallback mais inteligente baseado em ranges amplos
        if 430 <= comprimento_onda <= 510:
            return 'azul'
        elif 490 <= comprimento_onda <= 580:
            return 'verde'
        elif 570 <= comprimento_onda <= 630:
            return 'laranja'
        elif 620 <= comprimento_onda <= 750:
            return 'vermelho'

        return None
    def detectar_black_film_simulado(self, posicoes, comprimentos_onda):
        """
        ⚠️ CORREÇÃO: Black film SEMPRE é a PRIMEIRA região
        """
        if len(comprimentos_onda) < 10:
            return None

        # 1. DEFINIR REGIÃO INICIAL FIXA para black film
        # Black film sempre está no início (primeiros 0.5-1.0 cm)
        limite_black_film_cm = self.config['tamanho_janela_black_film']

        # Encontrar índices dentro da região inicial
        indices_black_film = np.where(posicoes <= limite_black_film_cm)[0]

        # ⚠️ CORREÇÃO: Verificar se o array não está vazio usando .size
        if indices_black_film.size == 0:
            # Se não encontrou pontos na região, usar primeiros 10% dos pontos
            num_pontos_black_film = max(5, len(comprimentos_onda) // 10)
            indices_black_film = np.arange(num_pontos_black_film)

        # Garantir que temos pelo menos alguns pontos
        if indices_black_film.size == 0:
            indices_black_film = np.array([0])

        # 2. Calcar características do black film
        pos_inicio = posicoes[indices_black_film[0]]
        pos_fim = posicoes[indices_black_film[-1]]
        pos_media = (pos_inicio + pos_fim) / 2

        comprimentos_black_film = comprimentos_onda[indices_black_film]
        comprimento_medio = np.mean(comprimentos_black_film)
        idx_representativo = indices_black_film[len(indices_black_film) // 2]
        comprimento_representativo = comprimentos_onda[idx_representativo]

        print(f"      🖤 Black film: {pos_inicio:.2f}-{pos_fim:.2f}cm, λ={comprimento_representativo:.1f}nm")

        return {
            'tipo': 'black_film',
            'label': 'Black Film',
            'posicao_inicio': pos_inicio,
            'posicao_fim': pos_fim,
            'posicao_representativa': pos_media,
            'comprimento_onda_medio': comprimento_medio,
            'comprimento_onda_representativo': comprimento_representativo,
            'indices': indices_black_film.tolist(),
            'tamanho_regiao': len(indices_black_film)
        }

    def detectar_cores_apos_black_film(self, posicoes, comprimentos_onda, inicio_cores, is_simulado=False):
        """
        CORREÇÃO: Considera TODOS os pontos extremos (picos e vales) como faixas de cor
        """
        if inicio_cores >= len(comprimentos_onda) - 5:
            return []

        # Trabalhar apenas com a região após black film
        posicoes_cores = posicoes[inicio_cores:]
        comprimentos_cores = comprimentos_onda[inicio_cores:]

        if len(comprimentos_cores) < 10:
            return []

        # 1. SUAVIZAR
        try:
            window_size = min(11, len(comprimentos_cores) // 10 * 2 + 1)
            if window_size % 2 == 0:
                window_size += 1
            comprimentos_suavizados = savgol_filter(comprimentos_cores, window_size, 3)
        except:
            comprimentos_suavizados = comprimentos_cores

        # 2. DETECTAR TODOS OS PICOS E VALES - ⚠️ CORREÇÃO: PARÂMETROS MAIS SENSÍVEIS
        features_detectadas = []

        # Detectar PICOS (máximos locais) - PARÂMETROS MAIS SENSÍVEIS
        try:
            picos, propriedades_picos = find_peaks(
                comprimentos_suavizados,
                prominence=2,  # ⚠️ REDUZIDO: detecta picos menores
                distance=8,  # ⚠️ REDUZIDO: detecta picos mais próximos
                height=400  # ⚠️ REDUZIDO: altura mínima menor
            )
            for pico in picos:
                features_detectadas.append({
                    'tipo': 'pico',
                    'indice': pico,
                    'valor': comprimentos_suavizados[pico],
                    'comprimento_onda': comprimentos_cores[pico],
                    'posicao': posicoes_cores[pico]
                })
        except Exception as e:
            print(f"      ⚠️ Erro detectando picos: {e}")

        # Detectar VALES (mínimos locais) - PARÂMETROS MAIS SENSÍVEIS
        try:
            # Inverter o sinal para detectar mínimos como máximos
            comprimentos_invertidos = -comprimentos_suavizados
            vales, propriedades_vales = find_peaks(
                comprimentos_invertidos,
                prominence=2,  # ⚠️ REDUZIDO: detecta vales menores
                distance=8  # ⚠️ REDUZIDO: detecta vales mais próximos
            )
            for vale in vales:
                features_detectadas.append({
                    'tipo': 'vale',
                    'indice': vale,
                    'valor': comprimentos_suavizados[vale],
                    'comprimento_onda': comprimentos_cores[vale],
                    'posicao': posicoes_cores[vale]
                })
        except Exception as e:
            print(f"      ⚠️ Erro detectando vales: {e}")

        # Ordenar features por posição
        features_ordenadas = sorted(features_detectadas, key=lambda x: x['indice'])
        print(f"      📈 Features detectadas: {len(features_ordenadas)}")

        # 3. DETECTAR REGIÕES ESTÁVEIS (quando não há features)
        gradiente = np.gradient(comprimentos_suavizados)
        gradiente_abs = np.abs(gradiente)

        limiar_estabilidade = np.percentile(gradiente_abs, 30)  # ⚠️ MAIS SENSÍVEL
        limiar_estabilidade = max(limiar_estabilidade, 0.2)  # ⚠️ LIMIAR MAIS BAIXO

        mascara_estavel = gradiente_abs < limiar_estabilidade
        indices_estaveis = np.where(mascara_estavel)[0]

        # Agrupar regiões estáveis
        regioes_estaveis = []
        if len(indices_estaveis) > 0:
            regiao_atual = [indices_estaveis[0]]
            for i in range(1, len(indices_estaveis)):
                if indices_estaveis[i] == indices_estaveis[i - 1] + 1:
                    regiao_atual.append(indices_estaveis[i])
                else:
                    if len(regiao_atual) >= 2:  # ⚠️ REDUZIDO: regiões menores
                        regioes_estaveis.append(regiao_atual)
                    regiao_atual = [indices_estaveis[i]]
            if len(regiao_atual) >= 2:
                regioes_estaveis.append(regiao_atual)

        # 4. CRIAR FAIXAS A PARTIR DAS FEATURES - ⚠️ CORREÇÃO: TODAS AS FEATURES SÃO FAIXAS
        cores_detectadas = []

        # PRIMEIRO: Todas as features (picos e vales) viram faixas
        for i, feature in enumerate(features_ordenadas):
            idx = feature['indice']

            # Criar uma pequena região ao redor da feature (3 pontos)
            inicio = max(0, idx - 1)
            fim = min(len(comprimentos_cores) - 1, idx + 1)

            # Ajustar índices para posições absolutas
            inicio_abs = inicio + inicio_cores
            fim_abs = fim + inicio_cores

            pos_inicio = posicoes[inicio_abs]
            pos_fim = posicoes[fim_abs]
            pos_media = feature['posicao']  # ⚠️ USA A POSIÇÃO EXATA DA FEATURE

            comprimentos_regiao = comprimentos_onda[inicio_abs:fim_abs + 1]
            comprimento_medio = np.mean(comprimentos_regiao)
            comprimento_representativo = feature['comprimento_onda']

            # Classificar cor
            tipo_cor = self.classificar_cor(comprimento_representativo, is_simulado)

            if not tipo_cor:
                # Se não classificou, tentar classificar pelo comprimento médio
                tipo_cor = self.classificar_cor(comprimento_medio, is_simulado)
                if not tipo_cor:
                    tipo_cor = f"cor_{i + 1}"

            # Determinar label
            if feature['tipo'] == 'pico':
                label = f"{tipo_cor.capitalize()} (Pico)"
            else:
                label = f"{tipo_cor.capitalize()} (Vale)"

            cores_detectadas.append({
                'tipo': tipo_cor,
                'label': label,
                'posicao_inicio': pos_inicio,
                'posicao_fim': pos_fim,
                'posicao_representativa': pos_media,  # ⚠️ POSIÇÃO EXATA DA FEATURE
                'comprimento_onda_medio': comprimento_medio,
                'comprimento_onda_representativo': comprimento_representativo,
                'indices': list(range(inicio_abs, fim_abs + 1)),
                'tamanho_regiao': 3,
                'eh_pico': feature['tipo'] == 'pico',
                'eh_vale': feature['tipo'] == 'vale',
                'eh_feature': True
            })

        # SEGUNDO: Adicionar regiões estáveis que não estão próximas de features
        for regiao in regioes_estaveis:
            # Verificar se não está muito próximo de feature existente
            muito_proximo = False
            for feature in features_ordenadas:
                if any(abs(idx - feature['indice']) < 3 for idx in regiao):
                    muito_proximo = True
                    break

            if not muito_proximo and len(regiao) >= 2:
                # Ajustar índices para posições absolutas
                inicio_abs = regiao[0] + inicio_cores
                fim_abs = regiao[-1] + inicio_cores

                pos_inicio = posicoes[inicio_abs]
                pos_fim = posicoes[fim_abs]
                pos_media = (pos_inicio + pos_fim) / 2

                comprimentos_regiao = comprimentos_onda[inicio_abs:fim_abs + 1]
                comprimento_medio = np.mean(comprimentos_regiao)
                comprimento_representativo = comprimentos_onda[inicio_abs + len(regiao) // 2]

                # Classificar cor
                tipo_cor = self.classificar_cor(comprimento_representativo, is_simulado)
                if not tipo_cor:
                    tipo_cor = f"cor_estavel_{len(cores_detectadas) + 1}"

                cores_detectadas.append({
                    'tipo': tipo_cor,
                    'label': f"{tipo_cor.capitalize()} (Estável)",
                    'posicao_inicio': pos_inicio,
                    'posicao_fim': pos_fim,
                    'posicao_representativa': pos_media,
                    'comprimento_onda_medio': comprimento_medio,
                    'comprimento_onda_representativo': comprimento_representativo,
                    'indices': list(range(inicio_abs, fim_abs + 1)),
                    'tamanho_regiao': len(regiao),
                    'eh_pico': False,
                    'eh_vale': False,
                    'eh_feature': False
                })

        # Ordenar por posição
        cores_detectadas = sorted(cores_detectadas, key=lambda x: x['posicao_representativa'])

        # Log das cores detectadas
        for cor in cores_detectadas:
            tipo_feature = ""
            if cor['eh_pico']:
                tipo_feature = "📈 PICO"
            elif cor['eh_vale']:
                tipo_feature = "📉 VALE"
            else:
                tipo_feature = "➡️ ESTÁVEL"
            print(f"      🎨 {cor['tipo']}: {cor['posicao_representativa']:.2f}cm ({tipo_feature})")

        return cores_detectadas
    def detectar_faixas_estaveis(self, posicoes, comprimentos_onda, is_simulado=False):
        """
        ⚠️ CORREÇÃO: Detecção que garante black film primeiro, depois cores
        """
        if len(comprimentos_onda) < 10:
            return []

        faixas_detectadas = []

        # 1. SEMPRE DETECTAR BLACK FILM PRIMEIRO
        black_film = self.detectar_black_film_simulado(posicoes, comprimentos_onda)
        if black_film:
            faixas_detectadas.append(black_film)
            inicio_cores = black_film['indices'][-1] + 1
        else:
            # Se não detectou black film, começar do início
            inicio_cores = 0

        # 2. DETECTAR CORES APÓS BLACK FILM - ⚠️ CORREÇÃO: Passar is_simulado
        cores = self.detectar_cores_apos_black_film(posicoes, comprimentos_onda, inicio_cores, is_simulado)
        faixas_detectadas.extend(cores)

        return faixas_detectadas

    def analisar_todos_tempos(self):
        """
        Analisa todos os tempos - USA EXATAMENTE AS CORES DO GRÁFICO 3
        """
        print("🎯 ANALISANDO INCERTEZAS - CORES DO GRÁFICO 3")
        print("=" * 60)
        print("⚠️  ESTRATÉGIA CORRIGIDA:")
        print("   • Black film SEMPRE primeiro (região inicial)")
        print("   • Usar EXATAMENTE as cores detectadas no Gráfico 3")
        print("   • Comparar apenas cores que existem em AMBOS os conjuntos")
        print("-" * 60)

        resultados_completos = {}

        for tempo in [60, 70, 80, 90]:
            if tempo not in self.dados_reais or tempo not in self.dados_simulados:
                print(f"⚠️  Dados incompletos para t={tempo}s")
                continue

            print(f"🔍 Processando t={tempo}s...")

            # Dados reais
            dados_reais = self.dados_reais[tempo]
            if 'dados_completos' in dados_reais:
                dados_reais = dados_reais['dados_completos']

            # Dados simulados
            dados_sim = self.dados_simulados[tempo]

            # Detectar faixas (MESMO ALGORITMO DO GRÁFICO 3)
            print(f"   📊 Detectando faixas REAIS...")
            faixas_reais = self.detectar_faixas_estaveis(
                np.array(dados_reais['posicoes_cm']),
                np.array(dados_reais['comprimentos_onda_nm']),
                is_simulado=False
            )

            print(f"   📊 Detectando faixas SIMULADAS...")
            faixas_sim = self.detectar_faixas_estaveis(
                np.array(dados_sim['posicoes_cm']),
                np.array(dados_sim['comprimentos_onda_nm']),
                is_simulado=True
            )

            # EXTRAIR CORES ÚNICAS DE CADA CONJUNTO (igual ao Gráfico 4)
            cores_reais_unicas = []
            faixas_reais_unicas = []
            for faixa in faixas_reais:
                if faixa['tipo'] not in cores_reais_unicas:
                    cores_reais_unicas.append(faixa['tipo'])
                    faixas_reais_unicas.append(faixa)

            cores_sim_unicas = []
            faixas_sim_unicas = []
            for faixa in faixas_sim:
                if faixa['tipo'] not in cores_sim_unicas:
                    cores_sim_unicas.append(faixa['tipo'])
                    faixas_sim_unicas.append(faixa)

            print(f"   🎨 Cores detectadas:")
            print(f"      Real: {cores_reais_unicas}")
            print(f"      Simulado: {cores_sim_unicas}")

            # ENCONTRAR CORES EM COMUM
            cores_para_comparar = []
            for cor in cores_reais_unicas:
                if cor in cores_sim_unicas:
                    cores_para_comparar.append(cor)

            print(f"   🔄 Comparando: {cores_para_comparar}")

            # Calcular diferenças para cores em comum
            diferencas = {}
            pontos_reais = {}
            pontos_simulados = {}

            for cor in cores_para_comparar:
                # Encontrar faixa REAL correspondente
                faixa_real = next((f for f in faixas_reais_unicas if f['tipo'] == cor), None)
                # Encontrar faixa SIMULADA correspondente
                faixa_sim = next((f for f in faixas_sim_unicas if f['tipo'] == cor), None)

                if faixa_real and faixa_sim:
                    diferenca = faixa_sim['posicao_representativa'] - faixa_real['posicao_representativa']
                    diferencas[cor] = diferenca
                    pontos_reais[cor] = {
                        'posicao': faixa_real['posicao_representativa'],
                        'comprimento_onda': faixa_real['comprimento_onda_representativo']
                    }
                    pontos_simulados[cor] = {
                        'posicao': faixa_sim['posicao_representativa'],
                        'comprimento_onda': faixa_sim['comprimento_onda_representativo']
                    }

                    print(f"      ✅ {cor}: Real {faixa_real['posicao_representativa']:.2f}cm → "
                          f"Sim {faixa_sim['posicao_representativa']:.2f}cm (Δ={diferenca:.3f}cm)")
                else:
                    diferencas[cor] = None
                    print(f"      ❌ {cor}: Não encontrado")

            resultados_completos[tempo] = {
                'tempo': tempo,
                'faixas_reais': faixas_reais,
                'faixas_simuladas': faixas_sim,
                'pontos_reais': pontos_reais,
                'pontos_simulados': pontos_simulados,
                'diferencas_posicionais': diferencas
            }

        return resultados_completos

    def gerar_grafico_3_espectros_faixas(self, resultados_completos):
        """
        GERA GRÁFICO 3: Espectros + faixas de cor + pontos médios
        """
        print("📊 GERANDO GRÁFICO 3: Espectros com faixas de cor")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        cores_faixas = {
            'black_film': 'black',
            'violeta': 'purple',
            'azul': 'blue',
            'verde': 'green',
            'amarelo': 'orange',
            'laranja': 'darkorange',
            'vermelho': 'red'
        }

        for idx, tempo in enumerate([60, 70, 80, 90]):
            if tempo not in resultados_completos:
                continue

            ax = axes[idx]
            resultado = resultados_completos[tempo]

            # Dados reais
            dados_reais = self.dados_reais[tempo]
            if 'dados_completos' in dados_reais:
                dados_reais = dados_reais['dados_completos']

            # Dados simulados
            dados_sim = self.dados_simulados[tempo]

            # Plot espectros
            ax.plot(dados_reais['posicoes_cm'], dados_reais['comprimentos_onda_nm'],
                    'k-', linewidth=2, label='Experimental', alpha=0.8)
            ax.plot(dados_sim['posicoes_cm'], dados_sim['comprimentos_onda_nm'],
                    'r-', linewidth=2, label='Simulado', alpha=0.8)

            # Plot faixas e pontos médios
            for faixa_real in resultado['faixas_reais']:
                cor = cores_faixas.get(faixa_real['tipo'], 'gray')

                # Área da faixa
                ax.axvspan(faixa_real['posicao_inicio'], faixa_real['posicao_fim'],
                           alpha=0.2, color=cor)

                # Ponto médio real
                marker = 'D' if faixa_real.get('eh_pico', False) else 'o'
                ax.plot(faixa_real['posicao_representativa'], faixa_real['comprimento_onda_representativo'],
                        marker, color=cor, markersize=8, markeredgecolor='white', markeredgewidth=2)

            for faixa_sim in resultado['faixas_simuladas']:
                cor = cores_faixas.get(faixa_sim['tipo'], 'gray')

                # Ponto médio simulado
                marker = 'D' if faixa_sim.get('eh_pico', False) else 's'
                ax.plot(faixa_sim['posicao_representativa'], faixa_sim['comprimento_onda_representativo'],
                        marker, color=cor, markersize=8, markeredgecolor='white', markeredgewidth=1, alpha=0.8)

            ax.set_title(f't = {tempo}s\nEspectros com Faixas de Cor', fontsize=12, fontweight='bold')
            ax.set_xlabel('Posição (cm)')
            ax.set_ylabel('Comprimento de Onda (nm)')
            ax.grid(True, alpha=0.3)
            ax.legend(['Real', 'Simulado'], loc='upper right')
            ax.set_xlim(0, 4)
            ax.set_ylim(400, 700)

        plt.tight_layout()
        plt.suptitle(
            'GRÁFICO 3: Espectros com Faixas de Cor e Pontos Médios\nBlack Film (Primeiro) → Cores (Picos/Estáveis)',
            fontsize=16, fontweight='bold', y=0.98)

        return fig

    def gerar_grafico_4_pontos_medios_erro(self, resultados_completos):
        """
        GERA 4 GRÁFICOS: Usando toda a área disponível
        """
        print("📊 GERANDO 4 GRÁFICOS: Ajustando limites do eixo X")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Cores para as features
        cores_ordem = ['black', 'purple', 'blue', 'green', 'orange', 'darkorange', 'red']

        for idx, tempo in enumerate([60, 70, 80, 90]):
            if tempo not in resultados_completos:
                continue

            ax = axes[idx]
            resultado = resultados_completos[tempo]

            # EXTRAIR CORES ÚNICAS EM ORDEM DE APARIÇÃO
            cores_reais_ordenadas = []
            posicoes_reais = []
            for faixa in resultado['faixas_reais']:
                if faixa['tipo'] not in cores_reais_ordenadas:
                    cores_reais_ordenadas.append(faixa['tipo'])
                    posicoes_reais.append(faixa['posicao_representativa'])

            cores_sim_ordenadas = []
            posicoes_sim = []
            for faixa in resultado['faixas_simuladas']:
                if faixa['tipo'] not in cores_sim_ordenadas:
                    cores_sim_ordenadas.append(faixa['tipo'])
                    posicoes_sim.append(faixa['posicao_representativa'])

            # SEPARAR BLACK FILM DAS DEMAIS CORES
            black_film_real = None
            outras_cores_reais = []
            outras_posicoes_reais = []

            for i, (cor, pos) in enumerate(zip(cores_reais_ordenadas, posicoes_reais)):
                if cor == 'black_film':
                    black_film_real = pos
                else:
                    outras_cores_reais.append(cor)
                    outras_posicoes_reais.append(pos)

            black_film_sim = None
            outras_cores_sim = []
            outras_posicoes_sim = []

            for i, (cor, pos) in enumerate(zip(cores_sim_ordenadas, posicoes_sim)):
                if cor == 'black_film':
                    black_film_sim = pos
                else:
                    outras_cores_sim.append(cor)
                    outras_posicoes_sim.append(pos)

            # COMPARAR PELA ORDEM
            num_comparacoes = min(len(outras_cores_reais), len(outras_cores_sim))

            # Coletar TODAS as posições para calcular limites automáticos
            todas_posicoes = []

            # Plotar BLACK FILM (primeiro)
            labels_plot = []
            if black_film_real is not None and black_film_sim is not None:
                diferenca_black = black_film_sim - black_film_real
                todas_posicoes.extend([black_film_real, black_film_sim])

                # Black film na posição 0
                y_pos_black = 0

                ax.errorbar(black_film_real, y_pos_black,
                            xerr=self.INCERTEZA_REGUA, fmt='o', color='black',
                            markersize=10, capsize=6, label='Real',
                            alpha=0.8, markeredgecolor='white', markeredgewidth=2)

                ax.errorbar(black_film_sim, y_pos_black,
                            xerr=self.INCERTEZA_REGUA, fmt='s', color='black',
                            markersize=10, capsize=6, label='Simulado',
                            alpha=0.8, markeredgecolor='white', markeredgewidth=2)

                ax.plot([black_film_real, black_film_sim], [y_pos_black, y_pos_black],
                        'k-', alpha=0.4, linewidth=2)

                # Anotação do black film MAIS AFASTADA
                ax.text((black_film_real + black_film_sim) / 2, y_pos_black + 0.15,
                        f'Δ={diferenca_black:.2f}cm',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9,
                                  edgecolor='black'))

                labels_plot.append('Black Film')

            # Plotar DEMAIS CORES por ordem (1ª, 2ª, 3ª...)
            diferencas_tempo = []
            for i in range(num_comparacoes):
                ordem = i + 1
                pos_real = outras_posicoes_reais[i]
                pos_sim = outras_posicoes_sim[i]
                cor_real = outras_cores_reais[i]
                cor_sim = outras_cores_sim[i]

                diferenca = pos_sim - pos_real
                diferencas_tempo.append(diferenca)
                todas_posicoes.extend([pos_real, pos_sim])

                # Posição vertical em ordem crescente (1ª cor após black film)
                y_pos = i + 1  # Black film é 0, 1ª cor é 1, 2ª cor é 2, etc.
                cor_plot = cores_ordem[ordem] if ordem < len(cores_ordem) else 'gray'

                # Plot ponto REAL
                ax.errorbar(pos_real, y_pos,
                            xerr=self.INCERTEZA_REGUA, fmt='o', color=cor_plot,
                            markersize=10, capsize=6,
                            label='Real' if i == 0 and black_film_real is None else "",
                            alpha=0.8, markeredgecolor='white', markeredgewidth=2)

                # Plot ponto SIMULADO
                ax.errorbar(pos_sim, y_pos,
                            xerr=self.INCERTEZA_REGUA, fmt='s', color=cor_plot,
                            markersize=10, capsize=6,
                            label='Simulado' if i == 0 and black_film_real is None else "",
                            alpha=0.8, markeredgecolor='white', markeredgewidth=2)

                # Linha conectando
                ax.plot([pos_real, pos_sim], [y_pos, y_pos],
                        color=cor_plot, alpha=0.5, linewidth=1.5, linestyle='--')

                # Anotação da diferença MAIS AFASTADA
                ax.text((pos_real + pos_sim) / 2, y_pos + 0.15,
                        f'Δ={diferenca:.2f}cm',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9,
                                  edgecolor=cor_plot))

                labels_plot.append(f'{ordem}ª Cor')

            # CONFIGURAÇÕES DO GRÁFICO COM LIMITES AUTOMÁTICOS
            ax.set_xlabel('Posição (cm)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cor', fontsize=11, fontweight='bold')

            ax.set_title(f't = {tempo}s\nComparação por Ordem de Aparição',
                         fontsize=13, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='x')

            # Legenda apenas no primeiro gráfico
            if idx == 0:
                ax.legend(loc='upper right', fontsize=10)

            # Estatísticas
            if diferencas_tempo:
                media_diff = np.mean(diferencas_tempo)
                std_diff = np.std(diferencas_tempo)
                ax.text(0.02, 0.98,
                        f'Δ médio: {media_diff:.3f} cm\nσ: {std_diff:.3f} cm\nn: {len(diferencas_tempo)} cores',
                        transform=ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

            # AJUSTAR LIMITES DO EIXO X AUTOMATICAMENTE
            if todas_posicoes:
                x_min = min(todas_posicoes) - 0.2
                x_max = max(todas_posicoes) + 0.3
                ax.set_xlim(x_min, x_max)
            else:
                ax.set_xlim(-0.1, 4.5)  # Fallback

            # Ajustar limites do eixo Y
            total_altura = len(labels_plot)
            ax.set_ylim(-0.5, total_altura + 0.5)

            # Labels do eixo Y
            ax.set_yticks(range(len(labels_plot)))
            ax.set_yticklabels(labels_plot, fontsize=10)

        # Layout final
        plt.tight_layout()
        plt.suptitle('GRÁFICO 4: Comparação por Ordem de Aparição\n'
                     f'Barras de erro = ±{self.INCERTEZA_REGUA} cm (Incerteza da Régua)',
                     fontsize=16, fontweight='bold', y=0.98)

        return fig

    def executar_analise_completa(self):
        """
        Executa análise completa e gera GRÁFICOS 3 e 4
        """
        print("🎯 EXECUTANDO ANÁLISE COMPLETA DE INCERTEZAS")
        print("=" * 60)
        print("FLUXO: Black Film (Primeiro) → Cores (Picos/Estáveis) → Análise")
        print("-" * 60)

        # 1. Analisar todos os tempos
        resultados_completos = self.analisar_todos_tempos()

        if not resultados_completos:
            print("❌ Nenhum resultado válido encontrado")
            return None

        # 2. Gerar Gráfico 3
        fig_grafico3 = self.gerar_grafico_3_espectros_faixas(resultados_completos)

        # 3. Gerar Gráfico 4
        fig_grafico4 = self.gerar_grafico_4_pontos_medios_erro(resultados_completos)

        # 4. Calcular estatísticas gerais
        todas_diferencas = []
        for tempo, resultado in resultados_completos.items():
            for cor, diff in resultado['diferencas_posicionais'].items():
                if diff is not None and cor != 'black_film':
                    todas_diferencas.append(diff)

        estatisticas = {
            'geral': {
                'media': np.mean(todas_diferencas) if todas_diferencas else 0,
                'desvio_padrao': np.std(todas_diferencas) if todas_diferencas else 0,
                'minimo': np.min(todas_diferencas) if todas_diferencas else 0,
                'maximo': np.max(todas_diferencas) if todas_diferencas else 0,
                'n_amostras': len(todas_diferencas)
            },
            'por_feature': {}
        }

        print(f"\n✅ ANÁLISE CONCLUÍDA!")
        print(f"📊 RESULTADOS:")
        print(f"   • {len(resultados_completos)} tempos analisados")
        print(f"   • {estatisticas['geral']['n_amostras']} diferenças calculadas")
        print(
            f"   • Erro médio: {estatisticas['geral']['media']:.3f} ± {estatisticas['geral']['desvio_padrao']:.3f} cm")
        print(f"   • Range: [{estatisticas['geral']['minimo']:.3f}, {estatisticas['geral']['maximo']:.3f}] cm")

        return {
            'resultados_tempos': resultados_completos,
            'figuras': [fig_grafico3, fig_grafico4],
            'estatisticas': estatisticas
        }


def executar_analise_incertezas(dados_reais, dados_simulados, plotar_exemplo=True):
    """
    Função principal para análise de incertezas - FLUXO CORRETO

    Args:
        dados_reais: dict de dados experimentais {tempo: dados}
        dados_simulados: dict de dados simulados {tempo: dados}
        plotar_exemplo: se deve mostrar os gráficos

    Returns:
        dict com resultados completos e figuras dos Gráficos 3 e 4
    """
    print("🔬 INICIANDO ANÁLISE DE INCERTEZAS (FLUXO CORRETO)")
    print("=" * 60)
    print("🎯 ESTRATÉGIA: Black Film (Primeiro) → Cores (Picos/Estáveis)")
    print("-" * 60)

    # Verificar dados de entrada
    tempos_esperados = [60, 70, 80, 90]
    tempos_reais = [t for t in tempos_esperados if t in dados_reais]
    tempos_sim = [t for t in tempos_esperados if t in dados_simulados]

    print(f"📋 DADOS DE ENTRADA:")
    print(f"   • Reais: {len(tempos_reais)} tempos → {tempos_reais}")
    print(f"   • Simulados: {len(tempos_sim)} tempos → {tempos_sim}")

    if not tempos_reais or not tempos_sim:
        print("❌ Dados insuficientes para análise")
        return None

    analisador = AnalisadorIncertezas(dados_reais, dados_simulados)
    resultados = analisador.executar_analise_completa()

    if resultados and plotar_exemplo:
        print("\n🖼️  GRÁFICOS GERADOS:")
        print("   • Gráfico 3: Espectros com faixas de cor e pontos médios")
        print("   • Gráfico 4: Comparação de pontos médios com barras de erro")
        print("   ⚠️  Feche as janelas para continuar...")
        plt.show(block=True)

    return resultados


def carregar_dados_existentes(tempos=[60, 70, 80, 90]):
    """
    Carrega dados reais salvos - COMPATIBILIDADE
    """
    from analisador_foto import carregar_dados_existentes as carregar_fotos

    print("📂 CARREGANDO DADOS REAIS...")
    dados_reais = carregar_fotos(tempos=tempos)

    if not dados_reais:
        print("❌ Nenhum dado real encontrado.")
        print("   Execute main_analise_fotos() primeiro!")
        return None

    print(f"✅ Dados carregados para tempos: {list(dados_reais.keys())}")
    return dados_reais


if __name__ == "__main__":
    print("🔬 MÓDULO DE ANÁLISE DE INCERTEZAS - FLUXO CORRETO")
    print("=" * 60)
    print("Este módulo gera:")
    print("   • GRÁFICO 3: Espectros + faixas de cor + pontos médios")
    print("   • GRÁFICO 4: Comparação pontos médios com barras de erro")
    print("=" * 60)

    # Exemplo de uso
    dados_reais = carregar_dados_existentes()

    if dados_reais:
        print("⚠️  Para uso completo, forneça dados simulados para comparação")
        print("   Use: executar_analise_incertezas(dados_reais, dados_simulados)")