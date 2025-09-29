# =============================================================================
# SIMULACAO.PY - MODELOS FÍSICOS E CÁLCULOS
# =============================================================================

import numpy as np


def calculate_thickness(t, h0, alpha, beta):
    """Calcula a espessura do filme ao longo do tempo"""
    return h0 * np.exp(-alpha * t) - (beta / alpha) * (1 - np.exp(-alpha * t))


def reflected_intensity(thickness, wl, n_film):
    """Calcula a intensidade refletida por interferência"""
    delta_phi = (4 * np.pi * n_film * thickness) / wl
    return np.cos(delta_phi / 2) ** 2


def wavelength_to_color(wl, intensity):
    """Converte comprimento de onda para cor RGB"""
    if wl < 380e-9 or wl > 750e-9:
        return np.array([0, 0, 0])
    if wl < 440e-9:
        r = -(wl - 440e-9) / (440e-9 - 380e-9);
        g = 0.0;
        b = 1.0
    elif wl < 490e-9:
        r = 0.0;
        g = (wl - 440e-9) / (490e-9 - 440e-9);
        b = 1.0
    elif wl < 510e-9:
        r = 0.0;
        g = 1.0;
        b = -(wl - 510e-9) / (510e-9 - 490e-9)
    elif wl < 580e-9:
        r = (wl - 510e-9) / (580e-9 - 510e-9);
        g = 1.0;
        b = 0.0
    elif wl < 645e-9:
        r = 1.0;
        g = -(wl - 645e-9) / (645e-9 - 580e-9);
        b = 0.0
    else:
        r = 1.0;
        g = 0.0;
        b = 0.0

    rgb = np.array([r, g, b]) * intensity
    return np.clip(rgb, 0, 1)


def run_simulation(params):
    """Executa simulação completa do fenômeno de interferência"""
    # Parâmetros de tempo e posição
    times = np.linspace(params['t_initial'], 0, params['num_steps'])
    x_cm = np.linspace(0, 5, params['num_steps'])

    # Cálculo da espessura
    thicknesses = calculate_thickness(times, params['h0'], params['alpha'], params['beta'])
    thicknesses = np.maximum(thicknesses, 0)
    thicknesses_nm = thicknesses * 1e9

    # Cálculo das cores por interferência
    wavelengths = np.linspace(380e-9, 750e-9, 1000)
    colors_array = []

    for thick in thicknesses:
        blended_color = np.zeros(3)
        total_intensity = 0

        for wl in wavelengths:
            intensity = reflected_intensity(thick, wl, params['n_film'])
            color = wavelength_to_color(wl, intensity)
            blended_color += color * intensity
            total_intensity += intensity

        if total_intensity > 0:
            blended_color /= total_intensity

        colors_array.append(blended_color)

    return {
        'thickness_nm': thicknesses_nm,
        'colors_rgb': np.array(colors_array),
        'x_cm': x_cm
    }
