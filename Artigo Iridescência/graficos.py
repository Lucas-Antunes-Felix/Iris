import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#GRAFICO DO PADRAO DE CORES

def plot_color_map_only(colors_array, x_cm=None, num_colors=500, title="Mapa de Cor"):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import numpy as np

    if x_cm is None:
        x_cm = np.linspace(0, 5, len(colors_array))

    # Interpolação para suavizar cores
    interp_func = interp1d(np.linspace(0, 1, len(colors_array)), colors_array, axis=0, kind='linear')
    interp_colors = interp_func(np.linspace(0, 1, num_colors))

    # Interpolação do eixo Y
    x_interp = np.linspace(x_cm[0], x_cm[-1], num_colors)

    plt.figure(figsize=(10, 6))
    for i in range(num_colors - 1):
        plt.fill_betweenx(
            [x_interp[i], x_interp[i + 1]],
            0, 1,  # eixo X fictício para visualização do gradiente
            color=interp_colors[i],
            edgecolor='none'
        )

    plt.ylabel("cm")
    plt.title(title)
    plt.xticks([])  # remove rótulos do eixo X
    plt.ylim(x_interp[-1], x_interp[0])  # inverter eixo Y para corresponder ao gráfico completo
    plt.tight_layout()
    plt.show()


#GRAFICO DO PADRAO DE CORES COM A VARIACAO DA ESPESSURA

def plot_color_map_with_thickness(thickness_array, colors_array, x_cm=None, num_colors=500,
                                  title="Variação da Espessura do Filme e Mapa de Cor"):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import numpy as np

    # Garantir que a espessura não seja negativa
    thickness_array = np.maximum(thickness_array, 0)
    thickness_nm = thickness_array   # converter para nm

    if x_cm is None:
        x_cm = np.linspace(0, 5, len(thickness_array))

    # Interpolação das cores para suavização
    interp_func = interp1d(np.linspace(0, 1, len(colors_array)), colors_array, axis=0, kind='linear')
    interp_colors = interp_func(np.linspace(0, 1, num_colors))

    # Inverter verticalmente o mapa de cores
    interp_colors_inverted = interp_colors[::-1]
    x_interp = np.linspace(x_cm[0], x_cm[-1], num_colors)[::-1]

    plt.figure(figsize=(10, 6))

    # Plot do mapa de cores limitado pela curva de espessura
    for i in range(num_colors - 1):
        # Interpolação da espessura correspondente à posição Y
        y = x_interp[i]
        # Encontrar o índice mais próximo na curva
        idx = np.abs(x_cm - y).argmin()
        # Limite X = espessura correspondente à curva
        plt.fill_betweenx(
            [x_interp[i], x_interp[i + 1]],
            0, thickness_nm[idx],  # bloqueia mapa de cor à esquerda da curva
            color=interp_colors_inverted[i],
            edgecolor='none'
        )

    # Sobrepor a curva da espessura
    plt.plot(thickness_nm, x_cm, color='black', linewidth=2, label='Espessura do Filme')

    # Configurações dos eixos
    plt.xlabel("Espessura do Filme (nm)")
    plt.ylabel("cm")
    plt.title(title)
    plt.xlim(0, np.max(thickness_nm) * 1.05)
    plt.ylim(x_cm[0], x_cm[-1])
    plt.gca().invert_yaxis()  # mantém visual invertido
    plt.legend()
    plt.tight_layout()
    plt.show()
