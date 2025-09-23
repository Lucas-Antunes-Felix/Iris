from pickle import FALSE

from simulacao import run_simulation
from graficos import plot_color_map_with_thickness, plot_color_map_only

# Parâmetros
params = {
    'h0': 1000e-9,
    'alpha': 1.1e-2,
    'beta': 0.9e-8,
    'n_film': 1.3,
    'num_steps': 1000,
    't_initial': 85
}

# Flags para escolher quais gráficos mostrar
MAPA_DE_COR = False
MAPA_DE_COR_COM_ESPESSURA = True

results = run_simulation(params)

if MAPA_DE_COR:
    plot_color_map_only(
        results['colors_rgb'],  # array de cores
        x_cm=results['x_cm']    # eixo Y opcional
    )

if MAPA_DE_COR_COM_ESPESSURA:
    plot_color_map_with_thickness(
        results['thickness_nm'],
        results['colors_rgb'],
        results['x_cm']
    )