# =============================================================================
# GRAFICOS.PY - VISUALIZAÇÕES E PLOTS
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Configurar para mostrar gráficos imediatamente
plt.ion()


def plot_color_map_only(colors_array, x_cm=None, title="Mapa de Cor da Interferência"):
    """Plota apenas o padrão de cores da interferência"""

    # 🔥 CORREÇÃO: Detectar se title foi passado erroneamente como array
    if (title is not None and
            not isinstance(title, str) and
            hasattr(title, '__len__') and
            len(title) > 1):
        print(f"⚠️  Aviso: 'title' recebeu um array em vez de string. Usando título padrão.")
        title = f"Mapa de Cores - Tempo Final: {tempo}s"

    if x_cm is None:
        x_cm = np.linspace(0, 5, len(colors_array))

    # Suavização das cores
    num_colors = 500
    interp_func = interp1d(np.linspace(0, 1, len(colors_array)), colors_array, axis=0, kind='linear')
    interp_colors = interp_func(np.linspace(0, 1, num_colors))
    x_interp = np.linspace(x_cm[0], x_cm[-1], num_colors)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_colors - 1):
        ax.fill_betweenx(
            [x_interp[i], x_interp[i + 1]],
            0, 1,
            color=interp_colors[i],
            edgecolor='none'
        )

    ax.set_ylabel("Posição (cm)")
    ax.set_title(title, fontsize=14, fontweight='bold')  # 🎯 TÍTULO CORRETO
    ax.set_xticks([])
    ax.set_ylim(x_interp[-1], x_interp[0])
    plt.tight_layout()

    plt.show()
    plt.pause(0.1)

    return fig

def plot_color_map_with_thickness(thickness_array, colors_array, x_cm=None,
                                  title="Padrão de Cores e Espessura do Filme"):
    """Plota cores da interferência com curva de espessura"""
    if x_cm is None:
        x_cm = np.linspace(0, 5, len(thickness_array))

    thickness_array = np.maximum(thickness_array, 0)

    # Suavização
    num_colors = 500
    interp_func = interp1d(np.linspace(0, 1, len(colors_array)), colors_array, axis=0, kind='linear')
    interp_colors = interp_func(np.linspace(0, 1, num_colors))
    interp_colors_inverted = interp_colors[::-1]
    x_interp = np.linspace(x_cm[0], x_cm[-1], num_colors)[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mapa de cores
    for i in range(num_colors - 1):
        idx = np.abs(x_cm - x_interp[i]).argmin()
        ax.fill_betweenx(
            [x_interp[i], x_interp[i + 1]],
            0, thickness_array[idx],
            color=interp_colors_inverted[i],
            edgecolor='none'
        )

    # Curva de espessura
    ax.plot(thickness_array, x_cm, color='black', linewidth=2, label='Espessura do Filme')

    ax.set_xlabel("Espessura (nm)")
    ax.set_ylabel("Posição (cm)")
    ax.set_title(title)
    ax.set_xlim(0, np.max(thickness_array) * 1.05)
    ax.set_ylim(x_cm[0], x_cm[-1])
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    plt.show()
    plt.pause(0.1)

    return fig


def plotar_comprimento_onda_vs_posicao(resultados_analise, title="Comprimento de Onda vs Posição"):
    """Plota relação entre comprimento de onda e posição"""
    posicoes = resultados_analise['posicoes_cm']
    comprimentos_onda = resultados_analise['comprimentos_onda_nm']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(posicoes, comprimentos_onda, 'bo-', linewidth=2, markersize=4, alpha=0.7)
    ax.set_xlabel('Posição (cm)')
    ax.set_ylabel('Comprimento de Onda (nm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Faixas espectrais de referência
    faixas = [(380, 450, 'Violeta', 'violet'),
              (450, 495, 'Azul', 'blue'),
              (495, 570, 'Verde', 'green'),
              (570, 590, 'Amarelo', 'yellow'),
              (590, 620, 'Laranja', 'orange'),
              (620, 750, 'Vermelho', 'red')]

    for inicio, fim, nome, cor in faixas:
        ax.axhspan(inicio, fim, alpha=0.1, color=cor, label=nome)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()
    plt.pause(0.1)

    return fig


def analisar_cores_para_comprimento_onda(colors_array, x_cm=None, num_amostras=100):
    """Analisa cores e converte para comprimentos de onda"""
    if x_cm is None:
        x_cm = np.linspace(0, 5, len(colors_array))

    # Amostragem
    if len(colors_array) > num_amostras:
        indices = np.linspace(0, len(colors_array) - 1, num_amostras, dtype=int)
    else:
        indices = np.arange(len(colors_array))

    posicoes_analisadas = []
    comprimentos_onda = []

    for idx in indices:
        cor_rgb = colors_array[idx]
        lambda_dominante = _rgb_para_comprimento_onda(cor_rgb)

        posicoes_analisadas.append(x_cm[idx])
        comprimentos_onda.append(lambda_dominante)

    return {
        'posicoes_cm': np.array(posicoes_analisadas),
        'comprimentos_onda_nm': np.array(comprimentos_onda) * 1e9
    }


def _rgb_para_comprimento_onda(rgb, intensidade_max=1.0):
    """Função interna: converte RGB para comprimento de onda"""
    r, g, b = rgb

    if intensidade_max > 0:
        r_norm, g_norm, b_norm = r / intensidade_max, g / intensidade_max, b / intensidade_max
    else:
        r_norm, g_norm, b_norm = 0, 0, 0

    # Determinação da região espectral
    if r_norm > g_norm and r_norm > b_norm:
        if g_norm > 0.5:
            lambda_approx = 580e-9 + (645e-9 - 580e-9) * (1 - g_norm)
        else:
            lambda_approx = 645e-9 - (645e-9 - 580e-9) * r_norm
    elif g_norm > r_norm and g_norm > b_norm:
        if r_norm > 0.5:
            lambda_approx = 580e-9 - (580e-9 - 510e-9) * g_norm
        else:
            lambda_approx = 510e-9 + (580e-9 - 510e-9) * g_norm
    elif b_norm > r_norm and b_norm > g_norm:
        if g_norm > 0.5:
            lambda_approx = 490e-9 + (510e-9 - 490e-9) * b_norm
        else:
            lambda_approx = 440e-9 + (490e-9 - 440e-9) * b_norm
    else:
        lambda_approx = 580e-9 if r_norm > 0.5 else 480e-9

    return lambda_approx