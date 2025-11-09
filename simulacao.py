# =============================================================================
# SIMULACAO.PY - FÍSICA CORRETA COM VISUALIZAÇÃO PRÁTICA
# =============================================================================

import numpy as np


def calculate_thickness(t, h0, alpha, beta):
    """Calcula a espessura do filme ao longo do tempo"""
    thickness = h0 * np.exp(-alpha * t) - (beta / alpha) * (1 - np.exp(-alpha * t))
    return np.maximum(thickness, 1e-10)


def fabry_perot_reflectivity_correct(thickness, wl, n_film, n_air=1.0):
    """
    Função de Airy CORRETA mas com ajuste para visualização
    """
    # Coeficiente de Fresnel para interface ar-filme
    r = (n_air - n_film) / (n_air + n_film)
    R = np.abs(r) ** 2  # Reflectividade (~0.04 para n_film=1.375)

    # Coeficiente de finesse
    F = 4 * R / (1 - R) ** 2

    # Diferença de fase (eq. 6.38)
    delta = (4 * np.pi * n_film * thickness) / wl

    # Função de Airy para intensidade REFLETIDA
    I_reflected = F * np.sin(delta / 2) ** 2 / (1 + F * np.sin(delta / 2) ** 2)

    # CORREÇÃO: Para visualização, aumentar o contraste mantendo a física
    # A função original varia entre 0 e ~0.15 para R=0.04
    # Escalar para melhor visualização
    I_enhanced = I_reflected * 3.0  # Aumentar contraste

    return np.clip(I_enhanced, 0, 1)


def wavelength_to_rgb_visible(wl, intensity):
    """
    Conversão que GARANTE cores visíveis
    """
    wl_nm = wl * 1e9

    if wl_nm < 380 or wl_nm > 750:
        return np.array([0, 0, 0])

    # Curvas de sensibilidade mais saturadas
    r = np.exp(-0.5 * ((wl_nm - 620) / 45) ** 2) + 0.3 * np.exp(-0.5 * ((wl_nm - 700) / 30) ** 2)
    g = np.exp(-0.5 * ((wl_nm - 530) / 45) ** 2) + 0.2 * np.exp(-0.5 * ((wl_nm - 480) / 25) ** 2)
    b = np.exp(-0.5 * ((wl_nm - 470) / 40) ** 2) + 0.4 * np.exp(-0.5 * ((wl_nm - 420) / 20) ** 2)

    rgb = np.array([r, g, b])

    # Normalizar
    max_val = np.max(rgb)
    if max_val > 0:
        rgb = rgb / max_val

    # CORREÇÃO CRÍTICA: Garantir brilho mínimo
    # Mínimo 40% de brilho, máximo 100%
    rgb = rgb * (0.4 + 0.6 * intensity)

    # Aumentar saturação
    rgb = np.clip(rgb * 1.3, 0, 1)

    return rgb


def run_simulation(params):
    """
    Simulação com física correta e cores VISÍVEIS
    """
    h0 = params.get('h0', 16013.70e-9)
    alpha = params.get('alpha', 0.06)
    beta = params.get('beta', 1.02e-08)
    n_film = params.get('n_film', 1.375)
    num_steps = params.get('num_steps', 1000)
    t_initial = params.get('t_initial', 85)

    print(f"🔧 Simulação Fabry-Perot - Cores visíveis")
    print(f"   h0: {h0 * 1e9:.1f} nm, alpha: {alpha:.4f}, n_film: {n_film}")

    # Configuração
    times = np.linspace(t_initial, 0, num_steps)
    x_cm = np.linspace(0, 5, num_steps)
    thicknesses = calculate_thickness(times, h0, alpha, beta)
    thicknesses_nm = thicknesses * 1e9

    print(f"📊 Faixa de espessuras: {thicknesses_nm[0]:.0f} nm → {thicknesses_nm[-1]:.0f} nm")

    # Espectro visível - mais pontos para melhor qualidade
    wavelengths = np.linspace(400e-9, 700e-9, 80)

    colors_array = []

    print("🎨 Gerando cores visíveis...")

    for i, thickness in enumerate(thicknesses):
        if i % 200 == 0:
            print(f"   Progresso: {i}/{len(thicknesses)}")

        # Integrar sobre TODO o espectro
        integrated_color = np.zeros(3)
        total_intensity = 0

        for wl in wavelengths:
            # Intensidade física (já ajustada para visualização)
            intensity = fabry_perot_reflectivity_correct(thickness, wl, n_film)

            # Converter para RGB (já com brilho garantido)
            color = wavelength_to_rgb_visible(wl, intensity)

            # Integrar contribuição
            integrated_color += color * intensity
            total_intensity += intensity

        # Normalizar
        if total_intensity > 0:
            integrated_color /= total_intensity

        # CORREÇÃO FINAL: Garantir que não fique escuro
        if np.max(integrated_color) < 0.3:
            # Aumentar brilho para cores muito escuras
            integrated_color = np.clip(integrated_color * 1.8, 0, 1)

        colors_array.append(integrated_color)

    # Verificar brilho médio
    cores = np.array(colors_array)
    avg_brightness = np.mean(cores)
    print(f"✅ Simulação concluída! Brilho médio: {avg_brightness:.3f}")

    # Correção final se necessário
    if avg_brightness < 0.4:
        print("💡 Aplicando correção final de brilho...")
        cores = np.clip(cores * 1.5, 0, 1)

    return {
        'thickness_nm': thicknesses_nm,
        'colors_rgb': cores,
        'x_cm': x_cm
    }