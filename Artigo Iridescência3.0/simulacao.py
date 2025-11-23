# simulacao.py - VERSÃO COMPATÍVEL COM O RESTO DO CÓDIGO

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def calculate_thickness(t, h0, alpha, beta):
    """Calcula a espessura do filme ao longo do tempo - COMPATÍVEL"""
    thickness = h0 * np.exp(-alpha * t) - (beta / alpha) * (1 - np.exp(-alpha * t))
    return np.maximum(thickness, 1e-10)


def fabry_perot_reflectivity_correct(thickness, wl, n_film, n_air=1.0):
    """
    Função de Airy - reflectância física - COMPATÍVEL
    """
    # Coeficiente de Fresnel para interface ar-filme
    r = (n_air - n_film) / (n_air + n_film)
    R = np.abs(r) ** 2

    # Coeficiente de finesse
    F = 4 * R / (1 - R) ** 2

    # Diferença de fase
    delta = (4 * np.pi * n_film * thickness) / wl

    # Função de Airy para reflectância
    reflectance = F * np.sin(delta / 2) ** 2 / (1 + F * np.sin(delta / 2) ** 2)

    return np.clip(reflectance, 0, 1)


def physical_color_perception(reflectance_spectrum, wl_nm):
    """
    Converte espectro de reflectância em cor RGB - FÍSICA PURA
    """
    # Espectro LED - BALANCEAMENTO NATURAL
    R_led = np.exp(-0.5 * ((wl_nm - 630) / 12) ** 2)
    G_led = np.exp(-0.5 * ((wl_nm - 530) / 18) ** 2)
    B_led = np.exp(-0.5 * ((wl_nm - 460) / 15) ** 2)

    # 🔵 Azul real
    B_led = np.exp(-0.5 * ((wl_nm - 460) / 15) ** 2)

    # 🔴 Vermelho real
    R_led = np.exp(-0.5 * ((wl_nm - 630) / 12) ** 2)

    # 🟢 Verde REAL (estreito, sem cauda amarela)
    G_led = np.exp(-0.5 * ((wl_nm - 525) / 8) ** 2)

    # 🔪 Corte brutal da parte amarela
    long_wave_cut = np.where(wl_nm > 560, 0.02, 1.0)
    G_led *= long_wave_cut

    # 🧮 Pesos (finalmente funcionam)
    R_led *= 1
    G_led *= 1
    B_led *= 1

    # ✔ Não normalize o total_led
    total_led = R_led + G_led + B_led
    # Resposta do olho
    L_eye = np.exp(-0.5 * ((wl_nm - 570) / 35) ** 2)
    M_eye = np.exp(-0.5 * ((wl_nm - 540) / 30) ** 2)
    S_eye = np.exp(-0.5 * ((wl_nm - 440) / 25) ** 2)

    # Intensidade refletida
    intensity_total = reflectance_spectrum * total_led

    # 3 integrais separadas
    R_perceived = np.trapz(intensity_total * L_eye, wl_nm)
    G_perceived = np.trapz(intensity_total * M_eye, wl_nm)
    B_perceived = np.trapz(intensity_total * S_eye, wl_nm)

    rgb = np.array([R_perceived, G_perceived, B_perceived])

    # Normalização PURA
    max_val = np.max(rgb)
    if max_val > 0:
        rgb = rgb / max_val

    # Apenas brilho mínimo se necessário
    if np.mean(rgb) < 0.2:
        rgb = rgb * 1.3

    return np.clip(rgb, 0, 1)


def analisar_cores_para_comprimento_onda(colors_rgb, x_cm):
    """
    Converte cores RGB para comprimentos de onda para comparação - COMPATÍVEL
    """
    comprimentos_onda = []

    for rgb in colors_rgb:
        if isinstance(rgb, (list, tuple, np.ndarray)) and len(rgb) >= 3:
            r, g, b = rgb[0], rgb[1], rgb[2]

            # Converter RGB normalizado para comprimento de onda
            r_norm = r
            g_norm = g
            b_norm = b

            # Lógica de mapeamento RGB → comprimento de onda
            if b_norm > 0.7 and r_norm > 0.5 and g_norm < 0.3:
                lambda_approx = 400  # Violeta
            elif b_norm > 0.6 and g_norm < 0.4:
                lambda_approx = 450  # Azul
            elif g_norm > 0.5 and b_norm > 0.3:
                lambda_approx = 500  # Verde-azulado
            elif g_norm > 0.6:
                lambda_approx = 540  # Verde
            elif r_norm > 0.6 and g_norm > 0.5:
                lambda_approx = 580  # Amarelo
            elif r_norm > 0.7:
                lambda_approx = 620  # Vermelho/Laranja
            else:
                lambda_approx = 550  # Amarelo-esverdeado

            comprimentos_onda.append(lambda_approx)
        else:
            comprimentos_onda.append(550)

    # Suavizar
    comprimentos_onda = np.array(comprimentos_onda)
    try:
        window_size = min(31, len(comprimentos_onda) // 10 * 2 + 1)
        if window_size % 2 == 0:
            window_size += 1
        comprimentos_suavizados = signal.savgol_filter(comprimentos_onda, window_size, 3)
    except:
        comprimentos_suavizados = comprimentos_onda

    return {
        'posicoes_cm': x_cm.tolist(),
        'comprimentos_onda_nm': comprimentos_suavizados.tolist()
    }


def plot_analise_convolucao(params, reflectance_spectrum, wl_nm, thickness, colors_rgb):
    """
    Gera os 3 gráficos específicos de análise da convolução - VERTICAL
    """
    try:
        # Criar figura com 3 subplots VERTICAIS
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(f'Análise de Convolução - Espessura: {thickness * 1e9:.1f} nm',
                     fontsize=16, fontweight='bold')

        # =========================================================================
        # GRÁFICO 1: Reflectância vs Comprimento de Onda (Senoide)
        # =========================================================================
        ax1 = axes[0]

        # Usar APENAS a espessura atual (não múltiplas)
        reflectance = reflectance_spectrum
        ax1.plot(wl_nm, reflectance, 'r-', linewidth=3, label=f'{thickness * 1e9:.0f} nm')

        ax1.set_xlabel('Comprimento de Onda (nm)')
        ax1.set_ylabel('Reflectância')
        ax1.set_title('Espectro de Reflectância Fabry-Perot')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # =========================================================================
        # GRÁFICO 2: Emissão LED vs Resposta do Olho Humano
        # =========================================================================
        ax2 = axes[1]

        # Espectro LED
        R_led = np.exp(-0.5 * ((wl_nm - 630) / 12) ** 2)
        G_led = np.exp(-0.5 * ((wl_nm - 525) / 8) ** 2)
        B_led = np.exp(-0.5 * ((wl_nm - 460) / 15) ** 2)

        # Corte da parte amarela
        long_wave_cut = np.where(wl_nm > 560, 0.02, 1.0)
        G_led *= long_wave_cut

        # Resposta do olho humano (cones)
        L_eye = np.exp(-0.5 * ((wl_nm - 570) / 35) ** 2)  # Cone L (vermelho)
        M_eye = np.exp(-0.5 * ((wl_nm - 540) / 30) ** 2)  # Cone M (verde)
        S_eye = np.exp(-0.5 * ((wl_nm - 440) / 25) ** 2)  # Cone S (azul)

        # Plot LED - linhas sólidas
        ax2.plot(wl_nm, R_led, 'r-', linewidth=3, label='LED Vermelho', alpha=0.8)
        ax2.plot(wl_nm, G_led, 'g-', linewidth=3, label='LED Verde', alpha=0.8)
        ax2.plot(wl_nm, B_led, 'b-', linewidth=3, label='LED Azul', alpha=0.8)

        # Plot cones - linhas pontilhadas
        ax2.plot(wl_nm, L_eye, 'r--', linewidth=2, label='Cone L (570nm)', alpha=0.6)
        ax2.plot(wl_nm, M_eye, 'g--', linewidth=2, label='Cone M (540nm)', alpha=0.6)
        ax2.plot(wl_nm, S_eye, 'b--', linewidth=2, label='Cone S (440nm)', alpha=0.6)

        ax2.set_xlabel('Comprimento de Onda (nm)')
        ax2.set_ylabel('Intensidade Relativa')
        ax2.set_title('Espectro LED vs Resposta dos Cones Oculares')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.1)

        # =========================================================================
        # GRÁFICO 3: Convolução com Área de Cada Cor
        # =========================================================================
        ax3 = axes[2]

        # Calcular intensidade refletida por canal
        intensity_R = reflectance_spectrum * R_led
        intensity_G = reflectance_spectrum * G_led
        intensity_B = reflectance_spectrum * B_led

        # Plot intensidades com áreas
        ax3.fill_between(wl_nm, intensity_R, alpha=0.6, color='red', label='Área Vermelha')
        ax3.fill_between(wl_nm, intensity_G, alpha=0.6, color='green', label='Área Verde')
        ax3.fill_between(wl_nm, intensity_B, alpha=0.6, color='blue', label='Área Azul')

        # Linhas para mostrar as curvas
        ax3.plot(wl_nm, intensity_R, 'r-', linewidth=1, alpha=0.9)
        ax3.plot(wl_nm, intensity_G, 'g-', linewidth=1, alpha=0.9)
        ax3.plot(wl_nm, intensity_B, 'b-', linewidth=1, alpha=0.9)

        ax3.set_xlabel('Comprimento de Onda (nm)')
        ax3.set_ylabel('Intensidade Refletida')
        ax3.set_title('Convolução: Área de Cada Cor')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.show()

        return fig

    except Exception as e:
        print(f"⚠️  Erro ao plotar análise de convolução: {e}")
        return None

def run_simulation_with_convolucion_analysis(params):
    """
    Versão da simulação que inclui análise de convolução - COMPATÍVEL
    """
    try:
        # Executar simulação normal primeiro
        resultados = run_simulation(params)

        if resultados is None:
            print("❌ Simulação principal falhou")
            return None

        # Verificar se deve analisar convolução
        if params.get('analisar_convolucao', False):
            print("\n🔍 ANALISANDO CONVOLUÇÃO...")

            # Pegar uma espessura representativa (do meio do array)
            thicknesses = resultados['thickness_nm'] * 1e-9  # Converter para metros
            idx_representativo = len(thicknesses) // 2
            thickness_rep = thicknesses[idx_representativo]

            # Parâmetros para análise
            wl_nm = np.linspace(400, 700, 301)
            wl_m = wl_nm * 1e-9
            n_film = params.get('n_film', 1.375)

            # Calcular espectro de reflectância para espessura representativa
            reflectance_spectrum = np.array([fabry_perot_reflectivity_correct(thickness_rep, wl, n_film)
                                             for wl in wl_m])

            # Pegar cor correspondente
            cor_representativa = resultados['colors_rgb'][idx_representativo]

            # Gerar gráficos de análise
            fig_analise = plot_analise_convolucao(
                params, reflectance_spectrum, wl_nm, thickness_rep, cor_representativa
            )

            if fig_analise:
                print("✅ Gráficos de convolução gerados com sucesso!")
                # Opcional: salvar figura
                # fig_analise.savefig('analise_convolucao.png', dpi=300, bbox_inches='tight')

        return resultados

    except Exception as e:
        print(f"❌ Erro na simulação com análise de convolução: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simulation(params):
    """
    Função principal de simulação - FORMATO COMPATÍVEL
    """
    try:
        h0 = params.get('h0')
        alpha = params.get('alpha')
        beta = params.get('beta')
        n_film = params.get('n_film')
        num_steps = params.get('num_steps')
        t_initial = params.get('t_initial')

        print(f"🔧 Simulação Fabry-Perot - VERSÃO COMPATÍVEL")
        print(f"   h0: {h0 * 1e9:.1f} nm, alpha: {alpha:.4f}, n_film: {n_film}")

        # Grid temporal e espacial - COMPATÍVEL
        times = np.linspace(t_initial, 0, num_steps)
        x_cm = np.linspace(0, 5, num_steps)

        # Calcular espessuras - COMPATÍVEL
        thicknesses = calculate_thickness(times, h0, alpha, beta)
        thicknesses_nm = thicknesses * 1e9
        thicknesses_angstrom = thicknesses_nm * 10

        print(f"📊 Faixa de espessuras: {thicknesses_nm[0]:.0f} nm a {thicknesses_nm[-1]:.1f} nm")

        # Espectro visível
        wl_nm = np.linspace(400, 700, 301)
        wl_m = wl_nm * 1e-9

        colors_array = []
        print("🎨 Gerando cores (física pura)...")

        # ⚠️ NOVO: Flag para gerar análise de convolução apenas uma vez
        convolucao_gerada = False

        for i, thickness in enumerate(thicknesses):
            if i % 100 == 0:
                thickness_A = thickness * 1e10
                print(f"   Progresso: {i}/{len(thicknesses)} (espessura: {thickness_A:.1f} Å)")

            # Calcular espectro de reflectância
            reflectance_spectrum = np.array([fabry_perot_reflectivity_correct(thickness, wl, n_film)
                                             for wl in wl_m])

            # Converter para RGB - FÍSICA PURA
            color = physical_color_perception(reflectance_spectrum, wl_nm)
            colors_array.append(color)

            # ⚠️ NOVO: Gerar gráficos de convolução para ESPESSURA DE 10000 nm
            if (params.get('analisar_convolucao', False) and
                    not convolucao_gerada and
                    abs(thickness * 1e9 - 10000) < 500):  # ✅ MUDADO PARA 10000 nm

                print(f"📊 Gerando gráficos de convolução para espessura: {thickness * 1e9:.1f} nm")
                plot_analise_convolucao(params, reflectance_spectrum, wl_nm, thickness, color)
                convolucao_gerada = True  # Garante que gera apenas uma vez

        # Estatísticas finais - FORMATO COMPATÍVEL
        cores = np.array(colors_array)

        avg_r = np.mean(cores[:, 0])
        avg_g = np.mean(cores[:, 1])
        avg_b = np.mean(cores[:, 2])

        print(f"✅ Simulação concluída!")
        print(f"   Cores médias: R={avg_r:.3f}, G={avg_g:.3f}, B={avg_b:.3f}")

        # RETORNAR NO FORMATO ESPERADO PELO RESTO DO CÓDIGO
        resultados = {
            'thickness_nm': thicknesses_nm,
            'thickness_angstrom': thicknesses_angstrom,
            'colors_rgb': cores,
            'x_cm': x_cm,
            'times': times,
            'reached_20A': thicknesses_angstrom[-1] <= 20
        }

        print(f"   ✅ Retornando {len(cores)} cores no formato compatível")
        return resultados

    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        import traceback
        traceback.print_exc()
        return None


# Teste de compatibilidade
if __name__ == "__main__":
    print("🧪 TESTANDO SIMULAÇÃO COMPATÍVEL")

    params_teste = {
        'h0': 16013.70e-9,
        'alpha': 0.06,
        'beta': 1.02e-08,
        'n_film': 1.375,
        'num_steps': 500,
        't_initial': 90
    }

    resultado = run_simulation(params_teste)

    if resultado:
        print(f"✅ Simulação funcionando!")
        print(f"   Keys retornados: {list(resultado.keys())}")
        print(f"   Formato colors_rgb: {np.array(resultado['colors_rgb']).shape}")
        print(f"   Formato x_cm: {np.array(resultado['x_cm']).shape}")
    else:
        print("❌ Simulação falhou")


