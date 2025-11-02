# =============================================================================
# SIMULACAO.PY - MODELOS FÍSICOS E CÁLCULOS (CORRIGIDO)
# =============================================================================

import numpy as np


def calculate_thickness(t, h0, alpha, beta):
    """Calcula a espessura do filme ao longo do tempo"""
    return h0 * np.exp(-alpha * t) - (beta / alpha) * (1 - np.exp(-alpha * t))


def reflected_intensity(thickness, wl, n_film, n_air=1.0, theta=0):
    """Calcula a intensidade refletida com correções"""
    n1 = n_air
    n2 = n_film
    n3 = n_air

    # Ângulo no filme
    theta2 = np.arcsin(n1 * np.sin(theta) / n2)

    # Diferença de fase
    delta_phi = (4 * np.pi * n2 * thickness * np.cos(theta2)) / wl

    # Coeficientes de Fresnel
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)

    # Intensidade refletida (equação de interferência Fabry-Pérot)
    R1, R2 = r12 ** 2, r23 ** 2
    numerator = R1 + R2 + 2 * np.sqrt(R1 * R2) * np.cos(delta_phi)
    denominator = 1 + R1 * R2 + 2 * np.sqrt(R1 * R2) * np.cos(delta_phi)
    R_eff = numerator / denominator

    # CORREÇÃO: Garantir que a intensidade fique em uma faixa visível
    # Aplicar um fator de escala para melhorar a visibilidade
    R_eff = R_eff * 2.0  # Aumentar o contraste

    return np.clip(R_eff, 0, 1)


def wavelength_to_color(wl, intensity):
    """Converte comprimento de onda para cor RGB"""
    if wl < 380e-9 or wl > 750e-9:
        return np.array([0, 0, 0])

    # Mapeamento do espectro visível para RGB
    if wl < 440e-9:
        r = -(wl - 440e-9) / (440e-9 - 380e-9)
        g = 0.0
        b = 1.0
    elif wl < 490e-9:
        r = 0.0
        g = (wl - 440e-9) / (490e-9 - 440e-9)
        b = 1.0
    elif wl < 510e-9:
        r = 0.0
        g = 1.0
        b = -(wl - 510e-9) / (510e-9 - 490e-9)
    elif wl < 580e-9:
        r = (wl - 510e-9) / (580e-9 - 510e-9)
        g = 1.0
        b = 0.0
    elif wl < 645e-9:
        r = 1.0
        g = -(wl - 645e-9) / (645e-9 - 580e-9)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

   #Aplicar a intensidade corrigindo Gama
    rgb = np.array([r, g, b]) * (intensity ** 0.7)  # Gamma correction
    return np.clip(rgb, 0, 1)


def run_simulation(params):
    """Executa simulação completa com correções"""
    # Parâmetros de tempo e posição
    times = np.linspace(params['t_initial'], 0, params['num_steps'])
    x_cm = np.linspace(0, 5, params['num_steps'])

    # Cálculo da espessura
    thicknesses = calculate_thickness(times, params['h0'], params['alpha'], params['beta'])
    thicknesses = np.maximum(thicknesses, 1e-10)  # Evitar espessura zero
    thicknesses_nm = thicknesses * 1e9

    # Cálculo das cores por interferência
    wavelengths = np.linspace(380e-9, 750e-9, 500)  # Reduzir para melhor performance
    colors_array = []

    for i, thick in enumerate(thicknesses):
        blended_color = np.zeros(3)
        max_intensity = 0

        # CORREÇÃO: Encontrar a intensidade máxima primeiro
        intensities = []
        for wl in wavelengths:
            intensity = reflected_intensity(thick, wl, params['n_film'])
            intensities.append(intensity)

        max_intensity = max(intensities) if max(intensities) > 0 else 1

        # Misturar cores com normalização correta
        for wl, intensity in zip(wavelengths, intensities):
            if max_intensity > 0:
                normalized_intensity = intensity / max_intensity
            else:
                normalized_intensity = 0

            color = wavelength_to_color(wl, normalized_intensity)
            blended_color += color

        # Normalizar a cor final
        if len(wavelengths) > 0:
            blended_color /= len(wavelengths)

        # CORREÇÃO: Aumentar saturação e brilho
        blended_color = blended_color ** 0.8  # Aumentar contraste
        blended_color = np.clip(blended_color * 1.2, 0, 1)  # Aumentar brilho

        colors_array.append(blended_color)

    return {
        'thickness_nm': thicknesses_nm,
        'colors_rgb': np.array(colors_array),
        'x_cm': x_cm
    }