"""
==============================================================================
СИМУЛЯЦІЯ ЕЛАСТОМЕРНОЇ НАКЛАДКИ НА КОЛЕСО ВІЗКА ДЛЯ ПІДВИЩЕННЯ ТЕРТЯ НА ЛЬОДУ
==============================================================================

Модель охоплює:
  1. Контактна механіка Герца (циліндр-площина)
  2. Ефективний коефіцієнт тертя: адгезійна + гістерезисна + оральна складові
  3. Вплив рідинної плівки (мокрий лід) та дренажних каналів
  4. Температурна залежність (принцип ВЛФ)
  5. Вплив наповнювача (карбонізована кавова гуща)
  6. Критичний кут нахилу
  7. Знос за Арчардом
  8. Параметричний аналіз та 3D-візуалізація

Автор: Vladyslav / Генеровано Claude
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# КОЛЬОРОВА СХЕМА
# ─────────────────────────────────────────────────────────────────────────────
C_BG       = '#0D1117'
C_PANEL    = '#161B22'
C_GRID     = '#21262D'
C_TEXT     = '#E6EDF3'
C_TEXT2    = '#8B949E'
C_ACCENT1  = '#58A6FF'   # блакитний — основний
C_ACCENT2  = '#3FB950'   # зелений
C_ACCENT3  = '#FF7B72'   # червоний
C_ACCENT4  = '#D2A8FF'   # фіолетовий
C_ACCENT5  = '#FFA657'   # жовтогарячий
C_ICE      = '#CAE8FF'   # колір льоду
C_RUBBER   = '#2D6A4F'   # колір гуми

plt.rcParams.update({
    'figure.facecolor'  : C_BG,
    'axes.facecolor'    : C_PANEL,
    'axes.edgecolor'    : C_GRID,
    'axes.labelcolor'   : C_TEXT,
    'xtick.color'       : C_TEXT2,
    'ytick.color'       : C_TEXT2,
    'text.color'        : C_TEXT,
    'grid.color'        : C_GRID,
    'grid.alpha'        : 0.6,
    'lines.linewidth'   : 2.0,
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 9,
    'legend.facecolor'  : '#21262D',
    'legend.edgecolor'  : C_GRID,
    'legend.labelcolor' : C_TEXT,
})

# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 1: ФІЗИЧНІ ПАРАМЕТРИ
# ══════════════════════════════════════════════════════════════════════════════

class SleeveParameters:
    """Параметри накладки та умов контакту."""

    # ── Геометрія ──────────────────────────────────────────────────────────
    R_wheel   = 0.075       # радіус колеса, м
    t_sleeve  = 0.006       # товщина накладки, м (6 мм)
    h_tread   = 0.003       # глибина протектора, м (3 мм)
    p_channel = 0.010       # крок каналів, м (10 мм)
    w_channel = 0.002       # ширина каналу, м
    d_channel = 0.001       # глибина каналу, м (для дренажу)
    L_contact = 0.020       # довжина контактної зони (ширина колеса), м

    # ── Матеріал матриці ───────────────────────────────────────────────────
    E_matrix  = 5e6         # модуль Юнга гуми/TPU при кімн. темп., Па
    nu_rubber = 0.49        # коефіцієнт Пуасона еластомеру
    Tg        = 213.15      # температура склування (−60 °C), К
    tan_delta = 0.15        # тангенс кута механічних втрат (кімн. т.)

    # ── Наповнювач ─────────────────────────────────────────────────────────
    phi_min   = 0.00        # мін. об'ємна частка наповнювача
    phi_max   = 0.20        # макс. об'ємна частка наповнювача
    phi_base  = 0.10        # базова об'ємна частка
    E_filler  = 10e9        # модуль Юнга карбонізованих частинок, Па

    # ── Лід ────────────────────────────────────────────────────────────────
    E_ice     = 9.0e9       # модуль Юнга льоду, Па
    nu_ice    = 0.33        # коефіцієнт Пуасона льоду
    H_ice     = 1.5e8       # твердість льоду, Па (твердість Вікерса)
    tau_ice   = 0.1e6       # зсувна міцність льоду, Па

    # ── Навантаження та кінематика ─────────────────────────────────────────
    m_cart    = 30.0        # маса візка, кг
    g         = 9.81        # прискорення вільного падіння, м/с²
    F_N       = m_cart * g  # нормальна сила, Н

    # ── Рідинна плівка ─────────────────────────────────────────────────────
    eta_water = 1.8e-3      # динамічна в'язкість, Па·с (вода при 0 °C)
    h_film_dry = 1e-9       # товщина плівки, сухий лід, м
    h_film_wet = 5e-6       # товщина плівки, мокрий лід, м
    h_film_salt= 2e-6       # товщина плівки, соляний лід, м

    # ── Знос (Арчард) ──────────────────────────────────────────────────────
    k_wear    = 1e-7        # константа зносу Арчарда (безрозм.)

    # ── Посадка ────────────────────────────────────────────────────────────
    delta_fit = 0.0005      # інтерференційний натяг, м

P = SleeveParameters()


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 2: ОСНОВНІ МОДЕЛІ
# ══════════════════════════════════════════════════════════════════════════════

def composite_modulus(E_matrix, phi, E_filler=P.E_filler):
    """
    Ефективний модуль еластомерного композиту за формулою Ейнштейна–Гута
    (з поправками вищих порядків).

    E_c ≈ E_m * (1 + 2.5φ + 6.2φ²)
    """
    return E_matrix * (1.0 + 2.5 * phi + 6.2 * phi**2)


def hertz_contact_halfwidth(F_N, R, L, E_star):
    """
    Півширина контактної плями за Герцем для циліндра на площині.

    a = sqrt(4 * F_N * R / (π * L * E*))
    """
    return np.sqrt(4.0 * F_N * R / (np.pi * L * E_star))


def reduced_modulus(E1, nu1, E2, nu2):
    """
    Зведений модуль пружності.

    1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
    """
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)


def wlf_shift(T, Tg, C1=17.44, C2=51.6):
    """
    Зсув за принципом еквівалентності час–температура (ВЛФ).

    log(a_T) = -C1*(T - Tg) / (C2 + (T - Tg))
    """
    dT = T - Tg
    if np.isscalar(dT):
        if dT <= -C2 + 0.1:
            return 1e15
        return -C1 * dT / (C2 + dT)
    else:
        result = np.where(dT > -C2 + 0.1,
                          -C1 * dT / (C2 + dT),
                          1e15)
        return result


def E_complex_temperature(T, E_room=P.E_matrix, Tg=P.Tg, tan_d=P.tan_delta):
    """
    Температурна залежність модуля еластомеру.

    Спрощена модель: E(T) = E_room * exp(-α*(T - T_room)) + E_glassy/(1+exp((T-Tg)/ΔT))
    """
    T_room  = 293.15
    alpha   = 0.015     # температурний коефіцієнт, К⁻¹
    E_glass = 2.0e9     # модуль у склоподібному стані
    DT      = 10.0      # ширина переходу, К

    E_elastic = E_room * np.exp(-alpha * (T - T_room))
    E_glass_contrib = E_glass / (1.0 + np.exp((T - Tg) / DT))
    E_storage = E_elastic + E_glass_contrib

    log_aT = wlf_shift(T, Tg)
    # Нормований тангенс кута втрат — пік поблизу Tg
    peak = np.exp(-0.5 * ((T - (Tg + 20)) / 15)**2)
    tan_delta_T = tan_d * (1.0 + 2.0 * peak)

    E_loss = E_storage * tan_delta_T
    return E_storage, E_loss, tan_delta_T


def mu_adhesion(tau_s, sigma_n):
    """Адгезійна складова тертя."""
    return tau_s / (sigma_n + 1e-3)   # запобігаємо діленню на 0


def mu_hysteresis(E_loss, E_star, C_rms=1e-4, q_min=1e2, q_max=1e6):
    """
    Гістерезисна складова для еластомерів.

    μ_hyst ∝ ∫ q² C(q) E''(ω(q)) / |E(ω)|² dq

    Спрощена оцінка з фрактальним спектром C(q) ~ q^(-2H-2), H=0.8.
    """
    H_frac = 0.8  # показник Херста
    # Нормований інтеграл
    def integrand(q):
        C_q = C_rms**2 / (q**(2 * H_frac + 2) + 1e-30)
        return q**2 * C_q

    val, _ = quad(integrand, q_min, q_max, limit=100)
    mu_h = (E_loss / (E_star**2 + 1e-30)) * val * 1e6   # масштабний коефіцієнт
    return np.clip(mu_h, 0.0, 0.3)


def mu_ploughing(tau_s, sigma_n, phi_filler):
    """
    Оральна складова (ploughing) від жорстких асперитів наповнювача.

    μ_plough ≈ (τ_s / σ_n) * φ
    """
    return (tau_s / (sigma_n + 1e-3)) * phi_filler


def film_reduction(h_film, a_contact, U, eta, p_mean, E_star):
    """
    Зменшення ефективного тертя через рідинну плівку.

    Повне гідродинамічне змащення:  Λ = h_film / (Ra)
    Якщо Λ > 3 → режим повного гідродинамічного змащення (μ_film ≈ μ_viscous)
    """
    # Параметр Зоммерфельда
    S = U * eta / (p_mean * a_contact + 1e-30)
    # Число Стрибека
    lambda_ratio = h_film / (a_contact * 0.01 + 1e-12)

    if np.isscalar(lambda_ratio):
        if lambda_ratio > 3.0:
            mu_film = 6.0 * np.pi * eta * U / (p_mean * h_film + 1e-30)
        elif lambda_ratio > 1.0:
            # Змішаний режим
            mu_film = 0.3 * (lambda_ratio - 1.0) / 2.0 * 0.05
        else:
            mu_film = 0.0
    else:
        mu_film = np.where(lambda_ratio > 3.0,
                           6.0 * np.pi * eta * U / (p_mean * h_film + 1e-30),
                           np.where(lambda_ratio > 1.0,
                                    0.3 * (lambda_ratio - 1.0) / 2.0 * 0.05,
                                    0.0))
    return np.clip(mu_film, 0.0, 0.5)


def drainage_condition(h_film, d_channel, p_channel_param, U, eta, p_mean, E_star):
    """
    Перевірка умови ефективного дренажу:
    h/d << 1  та  U*η/(p*E*) << 1
    """
    cond1 = h_film / (d_channel + 1e-12)
    cond2 = U * eta / (p_mean * E_star + 1e-30)
    # Ефективність дренажу (0..1)
    drain_eff = 1.0 / (1.0 + 10.0 * cond1) * 1.0 / (1.0 + 1000.0 * cond2)
    return drain_eff, cond1, cond2


def mu_effective(T, phi, h_film, U=0.5,
                 mode='wet',
                 use_drainage=True):
    """
    Повна модель ефективного коефіцієнта тертя.

    μ_eff = μ_adh + μ_hyst + μ_plough - μ_film

    Параметри:
        T        — температура, К
        phi      — об'ємна частка наповнювача
        h_film   — товщина рідинної плівки, м
        U        — швидкість ковзання, м/с
        mode     — 'dry', 'wet', 'salt'
        use_drainage — чи враховувати дренажні канали
    """
    # Матеріальні параметри
    E_m, E_loss, tan_d = E_complex_temperature(T)
    E_c = composite_modulus(E_m, phi)
    E_star = reduced_modulus(E_c, P.nu_rubber, P.E_ice, P.nu_ice)

    # Контактна механіка
    a = hertz_contact_halfwidth(P.F_N, P.R_wheel, P.L_contact, E_star)
    A_contact = 2.0 * a * P.L_contact
    p_mean = P.F_N / (A_contact + 1e-12)
    sigma_n = p_mean

    # Зсувна напруга на межі розділу
    tau_s = P.tau_ice * (1.0 + 0.5 * phi)   # наповнювач трохи підвищує τ

    # Складові тертя
    mu_adh   = mu_adhesion(tau_s, sigma_n)
    mu_hyst  = mu_hysteresis(E_loss, E_star)
    mu_plou  = mu_ploughing(tau_s, sigma_n, phi)

    # Плівка
    if mode == 'dry':
        h = P.h_film_dry
    elif mode == 'salt':
        h = P.h_film_salt
    else:
        h = h_film

    mu_film = film_reduction(h, a, U, P.eta_water, p_mean, E_star)

    # Дренаж
    if use_drainage and mode != 'dry':
        drain_eff, c1, c2 = drainage_condition(
            h, P.d_channel, P.p_channel, U, P.eta_water, p_mean, E_star)
        mu_film *= (1.0 - drain_eff * 0.8)   # дренаж зменшує вплив плівки

    mu_eff = mu_adh + mu_hyst + mu_plou - mu_film
    return np.clip(mu_eff, 0.02, 2.0)


def critical_angle(mu):
    """Критичний кут нахилу площини (градуси)."""
    return np.degrees(np.arctan(np.clip(mu, 0.01, 10.0)))


def archard_wear(F_N, L_slide, H=P.H_ice, k=P.k_wear):
    """
    Обсяг зносу за законом Арчарда.

    V = k * F_N * L / H
    """
    return k * F_N * L_slide / H


def contact_pressure_distribution(F_N, R, L, E_star, n_points=200):
    """
    Розподіл контактного тиску Герца вздовж контактної зони.

    p(x) = p0 * sqrt(1 - (x/a)²)
    """
    a = hertz_contact_halfwidth(F_N, R, L, E_star)
    p0 = 2.0 * F_N / (np.pi * a * L)   # максимальний тиск
    x = np.linspace(-a, a, n_points)
    p = p0 * np.sqrt(np.maximum(1.0 - (x / a)**2, 0.0))
    return x, p, a, p0


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 3: ОБЧИСЛЕННЯ СІТОК ДЛЯ ГРАФІКІВ
# ══════════════════════════════════════════════════════════════════════════════

# Температурний діапазон
T_range = np.linspace(233.15, 273.15, 120)   # −40 °C … 0 °C

# Об'ємна частка наповнювача
phi_range = np.linspace(0.0, 0.20, 100)

# Швидкість
U_range = np.linspace(0.01, 2.0, 100)

# ── Температурні залежності матеріалу ─────────────────────────────────────
E_stor_arr = np.zeros(len(T_range))
E_loss_arr = np.zeros(len(T_range))
tan_d_arr  = np.zeros(len(T_range))

for i, T in enumerate(T_range):
    E_stor_arr[i], E_loss_arr[i], tan_d_arr[i] = E_complex_temperature(T)

# ── μ_eff vs температура (3 стани льоду) ──────────────────────────────────
mu_dry  = np.array([mu_effective(T, P.phi_base, P.h_film_dry,  mode='dry') for T in T_range])
mu_wet  = np.array([mu_effective(T, P.phi_base, P.h_film_wet,  mode='wet') for T in T_range])
mu_salt = np.array([mu_effective(T, P.phi_base, P.h_film_salt, mode='salt') for T in T_range])

# ── μ_eff vs об'ємна частка наповнювача ───────────────────────────────────
T_test = 263.15   # −10 °C

mu_phi_dry  = np.array([mu_effective(T_test, phi, P.h_film_dry,  mode='dry')  for phi in phi_range])
mu_phi_wet  = np.array([mu_effective(T_test, phi, P.h_film_wet,  mode='wet')  for phi in phi_range])
mu_phi_salt = np.array([mu_effective(T_test, phi, P.h_film_salt, mode='salt') for phi in phi_range])

# ── Критичний кут ─────────────────────────────────────────────────────────
theta_dry  = critical_angle(mu_dry)
theta_wet  = critical_angle(mu_wet)
theta_salt = critical_angle(mu_salt)

# ── Контактний тиск ───────────────────────────────────────────────────────
E_m_test, _, _ = E_complex_temperature(T_test)
E_c_test = composite_modulus(E_m_test, P.phi_base)
E_star_test = reduced_modulus(E_c_test, P.nu_rubber, P.E_ice, P.nu_ice)

x_contact, p_contact, a_val, p0_val = contact_pressure_distribution(
    P.F_N, P.R_wheel, P.L_contact, E_star_test)

# ── Знос при різних умовах ────────────────────────────────────────────────
L_dist = np.linspace(0, 5000, 300)   # пробіг, м
V_dry  = archard_wear(P.F_N, L_dist)
V_wet  = archard_wear(P.F_N, L_dist, k=P.k_wear * 1.5)   # мокрий лід — вищий знос
V_salt = archard_wear(P.F_N, L_dist, k=P.k_wear * 2.0)

# ── 2D-сітка: μ_eff(T, φ) ────────────────────────────────────────────────
T_grid  = np.linspace(243.15, 273.15, 60)
phi_grid= np.linspace(0.0, 0.20, 60)
TG, PG  = np.meshgrid(T_grid, phi_grid)
MU_GRID = np.zeros_like(TG)

for i in range(TG.shape[0]):
    for j in range(TG.shape[1]):
        MU_GRID[i, j] = mu_effective(TG[i, j], PG[i, j],
                                     P.h_film_wet, mode='wet')

# ── Модуль vs температура для різних φ ───────────────────────────────────
phi_vals = [0.0, 0.05, 0.10, 0.15, 0.20]
E_phi_dict = {}
for phi in phi_vals:
    E_arr = np.array([composite_modulus(E_complex_temperature(T)[0], phi)
                      for T in T_range])
    E_phi_dict[phi] = E_arr

# ── μ vs швидкість (ефект Зоммерфельда) ──────────────────────────────────
mu_U_dry  = np.array([mu_effective(T_test, P.phi_base, P.h_film_dry,  U=u, mode='dry')  for u in U_range])
mu_U_wet  = np.array([mu_effective(T_test, P.phi_base, P.h_film_wet,  U=u, mode='wet')  for u in U_range])
mu_U_nodrain = np.array([mu_effective(T_test, P.phi_base, P.h_film_wet, U=u,
                                      mode='wet', use_drainage=False) for u in U_range])

# ── Ефективність дренажу ──────────────────────────────────────────────────
h_range = np.logspace(-8, -4, 200)
sigma_n_arr = P.F_N / (2 * hertz_contact_halfwidth(P.F_N, P.R_wheel, P.L_contact, E_star_test)
                       * P.L_contact)
drain_eff_arr = np.array([
    drainage_condition(h, P.d_channel, P.p_channel,
                       0.5, P.eta_water, sigma_n_arr, E_star_test)[0]
    for h in h_range])

# ── Тиск посадки ──────────────────────────────────────────────────────────
delta_range = np.linspace(0.0001, 0.002, 100)
E_theta = E_m_test   # кільцевий модуль
p_fit = E_theta / P.R_wheel * delta_range

# ── Складові μ vs φ ───────────────────────────────────────────────────────
mu_adh_phi  = []
mu_hyst_phi = []
mu_plou_phi = []
mu_film_phi = []

for phi in phi_range:
    E_m, E_loss, tan_d = E_complex_temperature(T_test)
    E_c = composite_modulus(E_m, phi)
    E_s = reduced_modulus(E_c, P.nu_rubber, P.E_ice, P.nu_ice)
    a   = hertz_contact_halfwidth(P.F_N, P.R_wheel, P.L_contact, E_s)
    A   = 2 * a * P.L_contact
    p_m = P.F_N / (A + 1e-12)
    tau = P.tau_ice * (1.0 + 0.5 * phi)

    mu_adh_phi.append(mu_adhesion(tau, p_m))
    mu_hyst_phi.append(mu_hysteresis(E_loss, E_s))
    mu_plou_phi.append(mu_ploughing(tau, p_m, phi))
    mf = film_reduction(P.h_film_wet, a, 0.5, P.eta_water, p_m, E_s)
    drain_e, _, _ = drainage_condition(P.h_film_wet, P.d_channel, P.p_channel,
                                       0.5, P.eta_water, p_m, E_s)
    mu_film_phi.append(mf * (1.0 - drain_e * 0.8))

mu_adh_phi  = np.array(mu_adh_phi)
mu_hyst_phi = np.array(mu_hyst_phi)
mu_plou_phi = np.array(mu_plou_phi)
mu_film_phi = np.array(mu_film_phi)

# Температура по Цельсію для відображення
T_C = T_range - 273.15


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 4: ПОБУДОВА ВЕЛИКОГО ДАШБОРДУ
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(C_BG)

gs = gridspec.GridSpec(
    5, 4,
    figure=fig,
    hspace=0.52,
    wspace=0.40,
    left=0.07, right=0.97,
    top=0.95, bottom=0.04
)

# ─── Заголовок ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         'ЕЛАСТОМЕРНА НАКЛАДКА НА КОЛЕСО ВІЗКА — МОДЕЛЬ ТЕРТЯ НА ЛЬОДУ',
         ha='center', va='top',
         fontsize=16, fontweight='bold', color=C_ACCENT1)
fig.text(0.5, 0.963,
         'Карбонізований наповнювач · Мікрорельєф · Дренажні канали · Контактна механіка Герца',
         ha='center', va='top',
         fontsize=10, color=C_TEXT2)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 1: μ_eff vs Температура (3 стани льоду)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2])

ax1.plot(T_C, mu_dry,  color=C_ACCENT2, lw=2.5, label='Сухий лід')
ax1.plot(T_C, mu_wet,  color=C_ACCENT1, lw=2.5, label='Мокрий лід', linestyle='--')
ax1.plot(T_C, mu_salt, color=C_ACCENT3, lw=2.5, label='Соляний розчин', linestyle='-.')

ax1.axvline(-10, color=C_TEXT2, lw=1.0, linestyle=':', alpha=0.7)
ax1.text(-9.5, ax1.get_ylim()[1] * 0.02 if ax1.get_ylim()[1] > 0 else 0.02,
         '−10 °C', color=C_TEXT2, fontsize=8)

ax1.fill_between(T_C, mu_wet, mu_dry, alpha=0.10, color=C_ACCENT1)
ax1.set_xlabel('Температура (°C)')
ax1.set_ylabel('μ_eff')
ax1.set_title('Ефективний коефіцієнт тертя vs Температура\n(φ = 10%, з дренажем)',
              fontsize=10, color=C_TEXT)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)
ax1.set_xlim(T_C[0], T_C[-1])

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 2: Критичний кут нахилу
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 2:])

ax2.plot(T_C, theta_dry,  color=C_ACCENT2, lw=2.5, label='Сухий лід')
ax2.plot(T_C, theta_wet,  color=C_ACCENT1, lw=2.5, linestyle='--', label='Мокрий лід')
ax2.plot(T_C, theta_salt, color=C_ACCENT3, lw=2.5, linestyle='-.', label='Соляний розчин')

# Типові значення для звичайної гуми
ax2.axhline(12.0, color=C_TEXT2, lw=1.2, linestyle=':', alpha=0.8)
ax2.text(T_C[-1] - 8, 12.5, 'Стандартна гума', color=C_TEXT2, fontsize=8)

ax2.fill_between(T_C, theta_wet, theta_dry, alpha=0.10, color=C_ACCENT2)
ax2.set_xlabel('Температура (°C)')
ax2.set_ylabel('θ_c (градуси)')
ax2.set_title('Критичний кут нахилу θ_c = arctan(μ_eff)\n(φ = 10%, з дренажем)',
              fontsize=10, color=C_TEXT)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)
ax2.set_xlim(T_C[0], T_C[-1])

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 3: Модуль пружності vs Температура
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, :2])

colors_phi = [C_ACCENT1, C_ACCENT2, C_ACCENT4, C_ACCENT5, C_ACCENT3]
for phi, col in zip(phi_vals, colors_phi):
    ax3.semilogy(T_C, E_phi_dict[phi] / 1e6, color=col, lw=2.0,
                 label=f'φ = {phi*100:.0f}%')

ax3.axvline(P.Tg - 273.15, color=C_ACCENT3, lw=1.5, linestyle='--', alpha=0.8)
ax3.text(P.Tg - 273.15 + 1, 1e3, 'T_g', color=C_ACCENT3, fontsize=9, fontweight='bold')

ax3.set_xlabel('Температура (°C)')
ax3.set_ylabel('E_c (МПа, лог. шкала)')
ax3.set_title('Ефективний модуль еластомерного\nкомпозиту vs Температура',
              fontsize=10, color=C_TEXT)
ax3.legend(fontsize=7.5, ncol=2)
ax3.grid(True, alpha=0.4, which='both')
ax3.set_xlim(T_C[0], T_C[-1])

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 4: Складові μ_eff vs об'ємна частка
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2:])

phi_pct = phi_range * 100

ax4.stackplot(phi_pct,
              mu_adh_phi,
              mu_hyst_phi,
              mu_plou_phi,
              labels=['μ_адг (адгезія)', 'μ_гіст (гістерезис)', 'μ_орал (оральний)'],
              colors=[C_ACCENT1 + '99', C_ACCENT4 + '99', C_ACCENT2 + '99'])

ax4.plot(phi_pct, mu_adh_phi + mu_hyst_phi + mu_plou_phi - mu_film_phi,
         color=C_ACCENT5, lw=2.5, label='μ_eff (з вирахуванням плівки)', zorder=5)
ax4.plot(phi_pct, mu_film_phi,
         color=C_ACCENT3, lw=2.0, linestyle='--', label='−μ_плівка', zorder=5)

ax4.set_xlabel('Об\'ємна частка φ (%)')
ax4.set_ylabel('Складові μ')
ax4.set_title('Складові ефективного коефіцієнта тертя\nvs Об\'ємна частка наповнювача (T = −10 °C)',
              fontsize=10, color=C_TEXT)
ax4.legend(fontsize=7.5, loc='upper left')
ax4.grid(True, alpha=0.4)
ax4.set_xlim(0, 20)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 5: Розподіл контактного тиску (Герц)
# ══════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, :2])

x_mm = x_contact * 1000   # м → мм
p_MPa = p_contact / 1e6   # Па → МПа

ax5.fill_between(x_mm, 0, p_MPa, color=C_ACCENT1, alpha=0.25)
ax5.plot(x_mm, p_MPa, color=C_ACCENT1, lw=2.5)

# Позначення
ax5.axvline( a_val * 1000, color=C_TEXT2, lw=1.0, linestyle=':', alpha=0.7)
ax5.axvline(-a_val * 1000, color=C_TEXT2, lw=1.0, linestyle=':', alpha=0.7)
ax5.annotate('', xy=(a_val * 1000, p0_val / 1e6 * 0.5),
             xytext=(-a_val * 1000, p0_val / 1e6 * 0.5),
             arrowprops=dict(arrowstyle='<->', color=C_ACCENT5, lw=1.5))
ax5.text(0, p0_val / 1e6 * 0.55, f'2a = {2*a_val*1000:.2f} мм',
         ha='center', fontsize=8.5, color=C_ACCENT5)
ax5.text(0, p0_val / 1e6 * 0.92,
         f'p₀ = {p0_val/1e6:.2f} МПа', ha='center', fontsize=8.5, color=C_ACCENT1)

ax5.set_xlabel('Положення вздовж контакту x (мм)')
ax5.set_ylabel('Контактний тиск p (МПа)')
ax5.set_title(f'Розподіл контактного тиску Герца\n(F_N = {P.F_N:.0f} Н, R = {P.R_wheel*100:.1f} см, T = −10 °C)',
              fontsize=10, color=C_TEXT)
ax5.grid(True, alpha=0.4)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 6: μ_eff vs Швидкість (з дренажем та без)
# ══════════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[2, 2:])

ax6.plot(U_range, mu_U_dry,     color=C_ACCENT2, lw=2.5, label='Сухий лід')
ax6.plot(U_range, mu_U_wet,     color=C_ACCENT1, lw=2.5, linestyle='--', label='Мокрий лід (з дренажем)')
ax6.plot(U_range, mu_U_nodrain, color=C_ACCENT3, lw=2.0, linestyle=':', label='Мокрий лід (без дренажу)')

ax6.fill_between(U_range, mu_U_wet, mu_U_nodrain,
                 alpha=0.15, color=C_ACCENT4, label='Ефект дренажу')

ax6.set_xlabel('Швидкість ковзання U (м/с)')
ax6.set_ylabel('μ_eff')
ax6.set_title('Вплив швидкості та дренажних каналів\nна μ_eff (T = −10 °C, φ = 10%)',
              fontsize=10, color=C_TEXT)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.4)
ax6.set_xlim(0, 2.0)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 7: Знос за Арчардом
# ══════════════════════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[3, :2])

ax7.plot(L_dist, V_dry  * 1e9, color=C_ACCENT2, lw=2.5, label='Сухий лід')
ax7.plot(L_dist, V_wet  * 1e9, color=C_ACCENT1, lw=2.5, linestyle='--', label='Мокрий лід (×1.5)')
ax7.plot(L_dist, V_salt * 1e9, color=C_ACCENT3, lw=2.5, linestyle='-.', label='Соляний розчин (×2.0)')

ax7.fill_between(L_dist, V_dry * 1e9, V_salt * 1e9, alpha=0.10, color=C_ACCENT3)

ax7_twin = ax7.twinx()
ax7_twin.plot(L_dist, V_dry * 1e9 / (P.t_sleeve * 1000),
              color=C_ACCENT4, lw=1.5, linestyle=':', alpha=0.7)
ax7_twin.set_ylabel('Відносний знос (×10⁻⁶ від товщини)', color=C_ACCENT4, fontsize=8)
ax7_twin.tick_params(axis='y', colors=C_ACCENT4)

ax7.set_xlabel('Пробіг L (м)')
ax7.set_ylabel('Обсяг зносу V (мм³)')
ax7.set_title('Знос накладки за законом Арчарда\n(k_wear, H_ice = const)',
              fontsize=10, color=C_TEXT)
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.4)
ax7.set_xlim(0, 5000)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 8: Ефективність дренажу vs товщина плівки
# ══════════════════════════════════════════════════════════════════════════════
ax8 = fig.add_subplot(gs[3, 2:])

ax8.semilogx(h_range * 1e6, drain_eff_arr * 100,
             color=C_ACCENT1, lw=2.5)
ax8.fill_between(h_range * 1e6, 0, drain_eff_arr * 100,
                 alpha=0.20, color=C_ACCENT1)

# Маркери типових умов
conditions = [
    (P.h_film_dry * 1e6,  'Сухий', C_ACCENT2),
    (P.h_film_salt * 1e6, 'Соляний', C_ACCENT3),
    (P.h_film_wet * 1e6,  'Мокрий', C_ACCENT1),
]
for h_mark, label, col in conditions:
    idx = np.argmin(np.abs(h_range * 1e6 - h_mark))
    eff = drain_eff_arr[idx] * 100
    ax8.plot(h_mark, eff, 'o', color=col, markersize=8, zorder=5)
    ax8.annotate(f'{label}\n{eff:.1f}%',
                 xy=(h_mark, eff),
                 xytext=(h_mark * 3, eff - 10),
                 fontsize=7.5, color=col,
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

ax8.set_xlabel('Товщина рідинної плівки h (мкм, лог. шкала)')
ax8.set_ylabel('Ефективність дренажу (%)')
ax8.set_title('Ефективність дренажних каналів\nvs Товщина рідинної плівки',
              fontsize=10, color=C_TEXT)
ax8.grid(True, alpha=0.4, which='both')
ax8.set_xlim(h_range[0] * 1e6, h_range[-1] * 1e6)
ax8.set_ylim(0, 105)

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 9: tan δ vs Температура
# ══════════════════════════════════════════════════════════════════════════════
ax9 = fig.add_subplot(gs[4, :2])

color_stor = C_ACCENT2
color_loss = C_ACCENT3
color_tan  = C_ACCENT4

ln1 = ax9.semilogy(T_C, E_stor_arr / 1e6, color=color_stor, lw=2.0,
                   label="E' (МПа)")
ln2 = ax9.semilogy(T_C, E_loss_arr / 1e6, color=color_loss, lw=2.0,
                   linestyle='--', label="E'' (МПа)")

ax9b = ax9.twinx()
ln3 = ax9b.plot(T_C, tan_d_arr, color=color_tan, lw=2.5,
                linestyle=':', label='tan δ')
ax9b.set_ylabel('tan δ', color=color_tan)
ax9b.tick_params(axis='y', colors=color_tan)

ax9.axvline(P.Tg - 273.15, color=C_ACCENT3, lw=1.2, linestyle='-.', alpha=0.7)
ax9.text(P.Tg - 273.15 + 1, ax9.get_ylim()[0] * 5 if ax9.get_ylim()[0] > 0 else 0.01,
         'T_g', color=C_ACCENT3, fontsize=8)

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax9.legend(lns, labs, fontsize=8, loc='lower right')
ax9.set_xlabel('Температура (°C)')
ax9.set_ylabel("E', E'' (МПа, лог. шкала)")
ax9.set_title("В'язко-пружні властивості: E', E'', tan δ\nvs Температура (ВЛФ-модель)",
              fontsize=10, color=C_TEXT)
ax9.grid(True, alpha=0.4, which='both')
ax9.set_xlim(T_C[0], T_C[-1])

# ══════════════════════════════════════════════════════════════════════════════
# ГРАФІК 10: Тиск посадки (інтерференційна посадка)
# ══════════════════════════════════════════════════════════════════════════════
ax10 = fig.add_subplot(gs[4, 2:])

ax10.plot(delta_range * 1000, p_fit / 1e6, color=C_ACCENT5, lw=2.5)
ax10.fill_between(delta_range * 1000, 0, p_fit / 1e6,
                  alpha=0.20, color=C_ACCENT5)

# Позначення базового натягу
idx_base = np.argmin(np.abs(delta_range - P.delta_fit))
ax10.plot(P.delta_fit * 1000, p_fit[idx_base] / 1e6,
          'o', color=C_ACCENT3, markersize=9, zorder=5)
ax10.annotate(f'δ = {P.delta_fit*1000:.1f} мм\np_i = {p_fit[idx_base]/1e6:.2f} МПа',
              xy=(P.delta_fit * 1000, p_fit[idx_base] / 1e6),
              xytext=(P.delta_fit * 1000 + 0.3, p_fit[idx_base] / 1e6 * 0.6),
              fontsize=8.5, color=C_ACCENT3,
              arrowprops=dict(arrowstyle='->', color=C_ACCENT3, lw=1.2))

ax10.set_xlabel('Інтерференційний натяг δ (мм)')
ax10.set_ylabel('Контактний тиск посадки p_i (МПа)')
ax10.set_title('Тиск інтерференційної посадки\np_i ≈ (E_θ / R) · δ',
               fontsize=10, color=C_TEXT)
ax10.grid(True, alpha=0.4)
ax10.set_xlim(0, delta_range[-1] * 1000)
ax10.set_ylim(0)

plt.savefig('/mnt/user-data/outputs/ice_friction_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ Дашборд збережено: ice_friction_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 5: 3D-ГРАФІКИ
# ══════════════════════════════════════════════════════════════════════════════

fig3d = plt.figure(figsize=(22, 10))
fig3d.patch.set_facecolor(C_BG)
fig3d.suptitle('3D-АНАЛІЗ: μ_eff(T, φ)  та  Критичний кут θ_c(T, φ)',
               color=C_ACCENT1, fontsize=14, fontweight='bold', y=0.98)

# ── 3D поверхня μ_eff(T, φ) ──────────────────────────────────────────────
ax3d1 = fig3d.add_subplot(121, projection='3d')
ax3d1.set_facecolor(C_PANEL)

T_C_grid = TG - 273.15
surf1 = ax3d1.plot_surface(T_C_grid, PG * 100, MU_GRID,
                           cmap='cool', edgecolor='none', alpha=0.85)
ax3d1.contourf(T_C_grid, PG * 100, MU_GRID,
               zdir='z', offset=MU_GRID.min() - 0.01,
               cmap='cool', alpha=0.4)

cbar1 = fig3d.colorbar(surf1, ax=ax3d1, shrink=0.5, pad=0.12)
cbar1.set_label('μ_eff', color=C_TEXT)
cbar1.ax.yaxis.set_tick_params(color=C_TEXT)
plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=C_TEXT)

ax3d1.set_xlabel('T (°C)', color=C_TEXT, labelpad=8)
ax3d1.set_ylabel('φ (%)', color=C_TEXT, labelpad=8)
ax3d1.set_zlabel('μ_eff', color=C_TEXT, labelpad=8)
ax3d1.set_title('μ_eff(T, φ)  Мокрий лід', color=C_TEXT, pad=12)
ax3d1.tick_params(colors=C_TEXT2)
ax3d1.xaxis.pane.fill = False
ax3d1.yaxis.pane.fill = False
ax3d1.zaxis.pane.fill = False
ax3d1.xaxis.pane.set_edgecolor(C_GRID)
ax3d1.yaxis.pane.set_edgecolor(C_GRID)
ax3d1.zaxis.pane.set_edgecolor(C_GRID)
ax3d1.view_init(elev=25, azim=-50)

# ── 3D поверхня θ_c(T, φ) ────────────────────────────────────────────────
ax3d2 = fig3d.add_subplot(122, projection='3d')
ax3d2.set_facecolor(C_PANEL)

THETA_GRID = np.degrees(np.arctan(MU_GRID))

surf2 = ax3d2.plot_surface(T_C_grid, PG * 100, THETA_GRID,
                           cmap='plasma', edgecolor='none', alpha=0.85)
ax3d2.contourf(T_C_grid, PG * 100, THETA_GRID,
               zdir='z', offset=THETA_GRID.min() - 0.5,
               cmap='plasma', alpha=0.4)

# Площина θ = 12° (стандартна гума)
T_plane = np.array([[T_C_grid.min(), T_C_grid.max()],
                    [T_C_grid.min(), T_C_grid.max()]])
P_plane = np.array([[0, 0], [20, 20]])
Z_plane = np.full_like(T_plane, 12.0)
ax3d2.plot_surface(T_plane, P_plane, Z_plane, alpha=0.18, color=C_TEXT2)
ax3d2.text(T_C_grid.min(), 10, 12.5, 'Стандарт\n(12°)', color=C_TEXT2, fontsize=7.5)

cbar2 = fig3d.colorbar(surf2, ax=ax3d2, shrink=0.5, pad=0.12)
cbar2.set_label('θ_c (°)', color=C_TEXT)
cbar2.ax.yaxis.set_tick_params(color=C_TEXT)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=C_TEXT)

ax3d2.set_xlabel('T (°C)', color=C_TEXT, labelpad=8)
ax3d2.set_ylabel('φ (%)', color=C_TEXT, labelpad=8)
ax3d2.set_zlabel('θ_c (°)', color=C_TEXT, labelpad=8)
ax3d2.set_title('Критичний кут θ_c(T, φ)  Мокрий лід', color=C_TEXT, pad=12)
ax3d2.tick_params(colors=C_TEXT2)
ax3d2.xaxis.pane.fill = False
ax3d2.yaxis.pane.fill = False
ax3d2.zaxis.pane.fill = False
ax3d2.xaxis.pane.set_edgecolor(C_GRID)
ax3d2.yaxis.pane.set_edgecolor(C_GRID)
ax3d2.zaxis.pane.set_edgecolor(C_GRID)
ax3d2.view_init(elev=25, azim=-50)

plt.savefig('/mnt/user-data/outputs/ice_friction_3d.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ 3D-графік збережено: ice_friction_3d.png")


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 6: СХЕМА КОНСТРУКЦІЇ (ПОПЕРЕЧНИЙ ПЕРЕРІЗ)
# ══════════════════════════════════════════════════════════════════════════════

fig_sch = plt.figure(figsize=(18, 7))
fig_sch.patch.set_facecolor(C_BG)
fig_sch.suptitle('СХЕМА НАКЛАДКИ: ПОПЕРЕЧНИЙ ПЕРЕРІЗ ТА ДЕТАЛЬ КОНТАКТУ',
                 color=C_ACCENT1, fontsize=13, fontweight='bold')

gs_sch = gridspec.GridSpec(1, 2, figure=fig_sch,
                           left=0.05, right=0.97,
                           top=0.88, bottom=0.06,
                           wspace=0.12)

# ── Панель A: Поперечний переріз колеса з накладкою ──────────────────────
ax_a = fig_sch.add_subplot(gs_sch[0])
ax_a.set_facecolor(C_BG)
ax_a.set_aspect('equal')

R_w = P.R_wheel * 100   # у см для зображення
t_s = P.t_sleeve * 100
h_t = P.h_tread * 100

# Обід колеса (сталь)
wheel_outer = Circle((0, 0), R_w, fill=True,
                      facecolor='#3A4759', edgecolor=C_TEXT2, lw=1.5)
wheel_inner = Circle((0, 0), R_w - 2.5, fill=True,
                      facecolor=C_BG, edgecolor='none')
ax_a.add_patch(wheel_outer)
ax_a.add_patch(wheel_inner)
ax_a.text(0, R_w - 1.3, 'Обід\n(сталь)', ha='center', va='center',
          fontsize=7.5, color=C_TEXT2)

# Еластомерна накладка (кільце)
theta_arr = np.linspace(0, 2 * np.pi, 500)

# Зовнішній радіус накладки з протектором (синусоїдальний мікрорельєф)
R_out_base = R_w + t_s
amp_tread  = h_t * 0.5
n_channels = 16   # кількість каналів по колу
R_tread = R_out_base + amp_tread * np.abs(np.sin(n_channels * theta_arr / 2))

x_out = R_tread * np.cos(theta_arr)
y_out = R_tread * np.sin(theta_arr)
x_in  = (R_w + 0.05) * np.cos(theta_arr)
y_in  = (R_w + 0.05) * np.sin(theta_arr)

sleeve_poly_x = np.concatenate([x_out, x_in[::-1]])
sleeve_poly_y = np.concatenate([y_out, y_in[::-1]])
sleeve = plt.Polygon(
    np.column_stack([sleeve_poly_x, sleeve_poly_y]),
    facecolor=C_RUBBER, edgecolor=C_ACCENT2, lw=1.0, alpha=0.85)
ax_a.add_patch(sleeve)
ax_a.text(0, R_w + t_s / 2, 'Еластомер\n(гума/TPU + наповнювач)',
          ha='center', va='center', fontsize=7, color=C_ACCENT2)

# Крижана поверхня
ice_y_top = -(R_out_base + amp_tread)
ax_a.fill_between([-R_w - 2, R_w + 2],
                  ice_y_top - 1.5, ice_y_top,
                  facecolor=C_ICE, alpha=0.35, label='Лід')
ax_a.fill_between([-R_w - 2, R_w + 2],
                  ice_y_top - 3.0, ice_y_top - 1.5,
                  facecolor='#97C8EB', alpha=0.20)
ax_a.text(R_w + 1.5, ice_y_top - 0.8, 'Льодова\nповерхня',
          ha='left', va='center', fontsize=7.5, color=C_ICE)

# Нормальна сила
ax_a.annotate('', xy=(0, R_out_base + 1.0), xytext=(0, R_out_base + 4.0),
              arrowprops=dict(arrowstyle='->', color=C_ACCENT3, lw=2.0))
ax_a.text(0.3, R_out_base + 2.5, f'F_N = {P.F_N:.0f} Н',
          color=C_ACCENT3, fontsize=8.5)

# Вісь
ax_a.plot(0, 0, '+', color=C_TEXT2, markersize=12, lw=2.0)

# Позначення
ax_a.annotate('', xy=(R_w, 0), xytext=(0, 0),
              arrowprops=dict(arrowstyle='->', color=C_TEXT2, lw=1.2))
ax_a.text(R_w / 2, 0.5, f'R = {R_w:.1f} см', ha='center', fontsize=7.5, color=C_TEXT2)

ax_a.set_xlim(-R_w - 4, R_w + 5)
ax_a.set_ylim(-R_out_base - 5, R_out_base + 6)
ax_a.set_title('A) Поперечний переріз колеса з накладкою', fontsize=10, color=C_TEXT)
ax_a.axis('off')

# Легенда
patches = [
    mpatches.Patch(facecolor='#3A4759', edgecolor=C_TEXT2, label='Сталевий обід'),
    mpatches.Patch(facecolor=C_RUBBER,  edgecolor=C_ACCENT2, label='Еластомерна накладка'),
    mpatches.Patch(facecolor=C_ICE,     alpha=0.5, label='Льодова поверхня'),
]
ax_a.legend(handles=patches, loc='upper left', fontsize=7.5,
            facecolor=C_PANEL, edgecolor=C_GRID)

# ── Панель B: Деталь контактної зони ──────────────────────────────────────
ax_b = fig_sch.add_subplot(gs_sch[1])
ax_b.set_facecolor(C_BG)

# Масштаб у мкм/мм для деталі
scale_x = 10   # мм
scale_y = 3    # мм

# Контактна зона — еластомер (деформований)
x_elast = np.linspace(-scale_x, scale_x, 400)
# Профіль нижньої поверхні еластомеру з каналами
p_ch_mm = P.p_channel * 1000
d_ch_mm = P.d_channel * 1000

def sleeve_profile(x_arr, p_ch, d_ch, h_tread_mm):
    """Профіль нижньої поверхні накладки з дренажними каналами."""
    y = np.zeros_like(x_arr)
    # Ребра між каналами
    for xi, x in enumerate(x_arr):
        xmod = x % p_ch
        if xmod < p_ch * 0.25 or xmod > p_ch * 0.75:
            y[xi] = 0.0   # плоска контактна поверхня
        else:
            y[xi] = -d_ch * np.sin(np.pi * (xmod - p_ch * 0.25) / (p_ch * 0.5))
    return y

y_sleeve_bottom = sleeve_profile(x_elast, p_ch_mm, d_ch_mm, h_t)

# Заповнення тіла еластомеру
y_sleeve_top = np.full_like(x_elast, 2.0)
ax_b.fill_between(x_elast, y_sleeve_bottom, y_sleeve_top,
                  facecolor=C_RUBBER, alpha=0.75)
ax_b.plot(x_elast, y_sleeve_bottom, color=C_ACCENT2, lw=1.8)

# Лід
ax_b.fill_between(x_elast, -1.5, 0.0,
                  facecolor=C_ICE, alpha=0.30)
ax_b.axhline(0.0, color=C_ICE, lw=1.5, alpha=0.7)

# Рідинна плівка (мокрий лід — між льодом та гумою)
h_film_vis = 0.08   # мм (для візуалізації)
ax_b.fill_between(x_elast,
                  np.maximum(y_sleeve_bottom, -h_film_vis), 0.0,
                  where=(y_sleeve_bottom > -0.05),
                  facecolor='#4FC3F7', alpha=0.60, label='Рідинна плівка')

# Дренажні канали — позначення
for x_ch in np.arange(-scale_x + p_ch_mm / 2, scale_x, p_ch_mm):
    ax_b.annotate('', xy=(x_ch, -d_ch_mm), xytext=(x_ch, -d_ch_mm - 0.5),
                  arrowprops=dict(arrowstyle='->', color='#4FC3F7', lw=1.0))

# Наповнювач (точки)
rng = np.random.default_rng(42)
n_particles = 80
x_part = rng.uniform(-scale_x, scale_x, n_particles)
y_part = rng.uniform(0.15, 1.80, n_particles)
mask = y_part > (sleeve_profile(x_part, p_ch_mm, d_ch_mm, h_t) + 0.05)
ax_b.scatter(x_part[mask], y_part[mask], s=6, color='#8B5E3C', alpha=0.75,
             zorder=4, label='Карбонізований наповнювач')

# Мікрорельєф (асперити)
for xi in np.arange(-scale_x + 0.5, scale_x, 0.8):
    h_asp = rng.uniform(0.02, 0.08)
    if sleeve_profile(np.array([xi]), p_ch_mm, d_ch_mm, h_t)[0] > -0.05:
        ax_b.plot([xi, xi + 0.15, xi + 0.3],
                  [sleeve_profile(np.array([xi]), p_ch_mm, d_ch_mm, h_t)[0],
                   sleeve_profile(np.array([xi]), p_ch_mm, d_ch_mm, h_t)[0] - h_asp,
                   sleeve_profile(np.array([xi]), p_ch_mm, d_ch_mm, h_t)[0]],
                  color=C_ACCENT2, lw=0.7, alpha=0.6)

# Розмірні позначення
ax_b.annotate('', xy=(-scale_x + p_ch_mm, 2.4),
              xytext=(-scale_x, 2.4),
              arrowprops=dict(arrowstyle='<->', color=C_ACCENT5, lw=1.5))
ax_b.text(-scale_x + p_ch_mm / 2, 2.55, f'Крок p={p_ch_mm:.0f} мм',
          ha='center', fontsize=7.5, color=C_ACCENT5)

ax_b.annotate('', xy=(scale_x - 0.3, 0.0),
              xytext=(scale_x - 0.3, -d_ch_mm),
              arrowprops=dict(arrowstyle='<->', color='#4FC3F7', lw=1.5))
ax_b.text(scale_x - 0.1, -d_ch_mm / 2,
          f'd={d_ch_mm*1000:.0f} мкм', ha='left', fontsize=7.5, color='#4FC3F7')

ax_b.annotate('', xy=(scale_x - 0.3, 0.0),
              xytext=(scale_x - 0.3, 2.0),
              arrowprops=dict(arrowstyle='<->', color=C_ACCENT4, lw=1.5))
ax_b.text(scale_x - 0.1, 1.0, f't={t_s*10:.0f} мм', ha='left', fontsize=7.5, color=C_ACCENT4)

# Контактна напруга (штрихові лінії)
x_stress = np.linspace(-a_val * 1000 * 0.7, a_val * 1000 * 0.7, 30)
for xs in x_stress:
    ys_len = 0.15 * np.sqrt(max(1.0 - (xs / (a_val * 1000 * 0.7))**2, 0.0))
    ax_b.plot([xs, xs], [0.0, ys_len], color=C_ACCENT3, lw=0.6, alpha=0.4)

# Напис "контактна зона"
ax_b.annotate('', xy=(-a_val * 1000 * 0.7, -0.9),
              xytext=(a_val * 1000 * 0.7, -0.9),
              arrowprops=dict(arrowstyle='<->', color=C_ACCENT1, lw=1.5))
ax_b.text(0, -1.1, f'Контактна зона 2a ≈ {2*a_val*1000:.2f} мм',
          ha='center', fontsize=8, color=C_ACCENT1)

ax_b.set_xlim(-scale_x - 0.5, scale_x + 2.5)
ax_b.set_ylim(-1.8, 3.0)
ax_b.set_xlabel('Ширина (мм)', color=C_TEXT)
ax_b.set_ylabel('Глибина (мм)', color=C_TEXT)
ax_b.set_title('Б) Деталь: контактна зона, дренаж, мікрорельєф', fontsize=10, color=C_TEXT)
ax_b.legend(loc='upper right', fontsize=7.5, facecolor=C_PANEL, edgecolor=C_GRID)
ax_b.grid(True, alpha=0.3)
ax_b.tick_params(colors=C_TEXT2)
ax_b.spines['bottom'].set_color(C_GRID)
ax_b.spines['left'].set_color(C_GRID)
ax_b.spines['top'].set_color(C_GRID)
ax_b.spines['right'].set_color(C_GRID)

# Мітки матеріалу
ax_b.text(0, 1.0, 'Еластомерна накладка\n(вулк. гума або TPU)',
          ha='center', va='center', fontsize=7.5, color=C_ACCENT2,
          bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PANEL, alpha=0.7,
                    edgecolor=C_ACCENT2, lw=0.8))
ax_b.text(0, -0.75, 'Лід', ha='center', va='center',
          fontsize=8, color=C_ICE,
          bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PANEL, alpha=0.7,
                    edgecolor=C_ICE, lw=0.8))

plt.savefig('/mnt/user-data/outputs/ice_friction_schematic.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ Схему збережено: ice_friction_schematic.png")


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 7: ПАРАМЕТРИЧНА ОПТИМІЗАЦІЯ (ТЕПЛОВА КАРТА)
# ══════════════════════════════════════════════════════════════════════════════

fig_opt = plt.figure(figsize=(20, 6))
fig_opt.patch.set_facecolor(C_BG)
fig_opt.suptitle('ПАРАМЕТРИЧНА ОПТИМІЗАЦІЯ: ТЕПЛОВІ КАРТИ μ_eff',
                 color=C_ACCENT1, fontsize=13, fontweight='bold')

gs_opt = gridspec.GridSpec(1, 3, figure=fig_opt,
                           left=0.06, right=0.97,
                           top=0.87, bottom=0.12,
                           wspace=0.35)

# ── Сітка T × φ для 3 режимів ─────────────────────────────────────────────
T_hm   = np.linspace(243.15, 273.15, 40)
phi_hm = np.linspace(0.0, 0.20, 40)
TH, PH = np.meshgrid(T_hm, phi_hm)

modes_list = ['dry', 'wet', 'salt']
titles_list = ['Сухий лід', 'Мокрий лід', 'Соляний розчин']
h_films     = [P.h_film_dry, P.h_film_wet, P.h_film_salt]

for k, (mode, title, hf) in enumerate(zip(modes_list, titles_list, h_films)):
    MH = np.zeros_like(TH)
    for i in range(TH.shape[0]):
        for j in range(TH.shape[1]):
            MH[i, j] = mu_effective(TH[i, j], PH[i, j], hf, mode=mode)

    ax_opt = fig_opt.add_subplot(gs_opt[k])
    im = ax_opt.contourf(TH - 273.15, PH * 100, MH,
                         levels=20, cmap='RdYlGn')

    # Контурна лінія μ = 0.3 (орієнтир для безпечного руху)
    cs = ax_opt.contour(TH - 273.15, PH * 100, MH,
                        levels=[0.25, 0.35, 0.45],
                        colors=['white', 'yellow', 'cyan'], linewidths=1.2, alpha=0.8)
    ax_opt.clabel(cs, fmt='%.2f', fontsize=7, colors=['white', 'yellow', 'cyan'])

    # Оптимальна точка
    idx_max = np.unravel_index(np.argmax(MH), MH.shape)
    ax_opt.plot(TH[idx_max] - 273.15, PH[idx_max] * 100,
                '*', color=C_ACCENT3, markersize=14, zorder=5)
    ax_opt.text(TH[idx_max] - 273.15 + 0.5, PH[idx_max] * 100 + 0.5,
                f'max={MH[idx_max]:.2f}', fontsize=7.5, color=C_ACCENT3)

    cbar = fig_opt.colorbar(im, ax=ax_opt)
    cbar.set_label('μ_eff', color=C_TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=C_TEXT2, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_TEXT2)

    ax_opt.set_xlabel('Температура (°C)', fontsize=9, color=C_TEXT)
    ax_opt.set_ylabel('Об\'ємна частка φ (%)', fontsize=9, color=C_TEXT)
    ax_opt.set_title(f'μ_eff(T, φ) — {title}', fontsize=10, color=C_TEXT)
    ax_opt.tick_params(colors=C_TEXT2, labelsize=8)
    for spine in ax_opt.spines.values():
        spine.set_edgecolor(C_GRID)

plt.savefig('/mnt/user-data/outputs/ice_friction_heatmaps.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ Теплові карти збережено: ice_friction_heatmaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 8: ПІДСУМКОВА ТАБЛИЦЯ ПАРАМЕТРІВ (ТЕКСТОВИЙ ГРАФІК)
# ══════════════════════════════════════════════════════════════════════════════

fig_tab = plt.figure(figsize=(14, 8))
fig_tab.patch.set_facecolor(C_BG)
ax_tab = fig_tab.add_subplot(111)
ax_tab.set_facecolor(C_BG)
ax_tab.axis('off')

fig_tab.suptitle('ЗВЕДЕНА ТАБЛИЦЯ: РОЗРАХУНКОВІ ХАРАКТЕРИСТИКИ НАКЛАДКИ',
                 color=C_ACCENT1, fontsize=13, fontweight='bold')

# Обчислення підсумкових значень
mu_eff_dry_m10  = mu_effective(263.15, P.phi_base, P.h_film_dry,  mode='dry')
mu_eff_wet_m10  = mu_effective(263.15, P.phi_base, P.h_film_wet,  mode='wet')
mu_eff_salt_m10 = mu_effective(263.15, P.phi_base, P.h_film_salt, mode='salt')

theta_dry_m10   = critical_angle(mu_eff_dry_m10)
theta_wet_m10   = critical_angle(mu_eff_wet_m10)
theta_salt_m10  = critical_angle(mu_eff_salt_m10)

E_m_m10, _, _  = E_complex_temperature(263.15)
E_c_m10         = composite_modulus(E_m_m10, P.phi_base)
E_star_m10      = reduced_modulus(E_c_m10, P.nu_rubber, P.E_ice, P.nu_ice)
a_m10           = hertz_contact_halfwidth(P.F_N, P.R_wheel, P.L_contact, E_star_m10)
p_mean_m10      = P.F_N / (2 * a_m10 * P.L_contact)
p_fit_base      = E_m_m10 / P.R_wheel * P.delta_fit

V_dry_1km  = archard_wear(P.F_N, 1000) * 1e9
V_wet_1km  = archard_wear(P.F_N, 1000, k=P.k_wear * 1.5) * 1e9
V_salt_1km = archard_wear(P.F_N, 1000, k=P.k_wear * 2.0) * 1e9

table_data = [
    ['ПАРАМЕТР', 'ЗНАЧЕННЯ', 'ОДИНИЦІ', 'ПРИМІТКА'],
    # Геометрія
    ['─── Геометрія накладки ───', '', '', ''],
    ['Товщина накладки t',       f'{P.t_sleeve*1000:.1f}',     'мм',   ''],
    ['Глибина протектора h_t',   f'{P.h_tread*1000:.1f}',      'мм',   ''],
    ['Крок каналів p',           f'{P.p_channel*1000:.0f}',    'мм',   ''],
    ['Глибина каналу d',         f'{P.d_channel*1000:.1f}',    'мм',   ''],
    ['Радіус колеса R',          f'{P.R_wheel*100:.1f}',       'см',   ''],
    # Матеріал
    ['─── Матеріал ───', '', '', ''],
    ['Модуль Юнга (−10 °C)',     f'{E_m_m10/1e6:.2f}',         'МПа',  ''],
    ['Модуль композиту (φ=10%)', f'{E_c_m10/1e6:.2f}',         'МПа',  '+Геnnштейн 2.5φ + 6.2φ²'],
    ['Зведений модуль E*',       f'{E_star_m10/1e9:.3f}',      'ГПа',  ''],
    ['Температура склування T_g',f'{P.Tg - 273.15:.0f}',       '°C',   ''],
    # Контакт
    ['─── Контактна механіка ───', '', '', ''],
    ['Нормальна сила F_N',       f'{P.F_N:.1f}',               'Н',    f'маса {P.m_cart:.0f} кг'],
    ['Півширина контакту a',     f'{a_m10*1000:.3f}',           'мм',   'Герц, T=−10°C, φ=10%'],
    ['Середній тиск p_mean',     f'{p_mean_m10/1e6:.3f}',      'МПа',  ''],
    # μ_eff
    ['─── Тертя при T = −10 °C ───', '', '', ''],
    ['μ_eff, сухий лід',         f'{mu_eff_dry_m10:.4f}',      '—',    ''],
    ['μ_eff, мокрий лід',        f'{mu_eff_wet_m10:.4f}',      '—',    'з дренажем'],
    ['μ_eff, соляний розчин',    f'{mu_eff_salt_m10:.4f}',     '—',    'з дренажем'],
    ['θ_c, сухий лід',           f'{theta_dry_m10:.2f}',       '°',    ''],
    ['θ_c, мокрий лід',          f'{theta_wet_m10:.2f}',       '°',    ''],
    ['θ_c, соляний розчин',      f'{theta_salt_m10:.2f}',      '°',    ''],
    # Знос
    ['─── Знос за 1 км (Арчард) ───', '', '', ''],
    ['V, сухий лід',             f'{V_dry_1km:.4f}',           'мм³',  f'k={P.k_wear:.0e}'],
    ['V, мокрий лід',            f'{V_wet_1km:.4f}',           'мм³',  'k×1.5'],
    ['V, соляний розчин',        f'{V_salt_1km:.4f}',          'мм³',  'k×2.0'],
    # Посадка
    ['─── Інтерференційна посадка ───', '', '', ''],
    ['Натяг δ',                  f'{P.delta_fit*1000:.2f}',    'мм',   ''],
    ['Тиск посадки p_i',         f'{p_fit_base/1e6:.3f}',      'МПа',  ''],
]

col_widths = [0.35, 0.15, 0.12, 0.35]
col_x      = [0.02, 0.38, 0.54, 0.67]
row_height = 0.032
y_start    = 0.95
fontsize_h = 9
fontsize_b = 8

for r_idx, row in enumerate(table_data):
    y_pos = y_start - r_idx * row_height
    is_header = (r_idx == 0)
    is_section = row[0].startswith('───')

    for c_idx, (cell, cx) in enumerate(zip(row, col_x)):
        if is_header:
            color = C_ACCENT1
            fw = 'bold'
            fs = fontsize_h
        elif is_section:
            color = C_ACCENT4
            fw = 'bold'
            fs = fontsize_b
        else:
            color = C_TEXT if c_idx < 2 else C_TEXT2
            fw = 'normal'
            fs = fontsize_b

        ax_tab.text(cx, y_pos, cell,
                    transform=ax_tab.transAxes,
                    fontsize=fs, color=color, fontweight=fw,
                    va='top')

    # Розділювальна лінія
    if is_header or is_section:
        lw  = 1.2 if is_header else 0.5
        col = C_ACCENT1 if is_header else C_ACCENT4
        line = plt.Line2D([0.01, 0.99], [y_pos - 0.005, y_pos - 0.005],
                          transform=ax_tab.transAxes,
                          color=col, lw=lw, alpha=0.6)
        ax_tab.add_line(line)

plt.savefig('/mnt/user-data/outputs/ice_friction_table.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("✅ Таблицю збережено: ice_friction_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# ФІНАЛЬНИЙ ЗВІТ У КОНСОЛЬ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  РЕЗУЛЬТАТИ РОЗРАХУНКУ НАКЛАДКИ (T = −10 °C, φ = 10%)")
print("═" * 60)
print(f"  Модуль еластомеру E_m    = {E_m_m10/1e6:.2f} МПа")
print(f"  Модуль композиту E_c     = {E_c_m10/1e6:.2f} МПа")
print(f"  Зведений модуль  E*      = {E_star_m10/1e9:.4f} ГПа")
print(f"  Півширина Герца  a       = {a_m10*1000:.4f} мм")
print(f"  Середній тиск   p_mean  = {p_mean_m10/1e6:.4f} МПа")
print(f"  μ_eff  (сухий лід)       = {mu_eff_dry_m10:.4f}")
print(f"  μ_eff  (мокрий лід)      = {mu_eff_wet_m10:.4f}")
print(f"  μ_eff  (соляний розчин)  = {mu_eff_salt_m10:.4f}")
print(f"  θ_c    (сухий лід)       = {theta_dry_m10:.2f}°")
print(f"  θ_c    (мокрий лід)      = {theta_wet_m10:.2f}°")
print(f"  θ_c    (соляний розчин)  = {theta_salt_m10:.2f}°")
print(f"  Знос за 1 км (сухий)     = {V_dry_1km:.5f} мм³")
print(f"  Тиск посадки p_i         = {p_fit_base/1e6:.3f} МПа")
print("═" * 60)
print("\n  Збережені файли:")
print("  📊 ice_friction_dashboard.png  — головний дашборд (10 графіків)")
print("  🌐 ice_friction_3d.png         — 3D-поверхні μ і θ")
print("  🔧 ice_friction_schematic.png  — схема конструкції")
print("  🗺  ice_friction_heatmaps.png   — теплові карти оптимізації")
print("  📋 ice_friction_table.png      — зведена таблиця параметрів")
print("═" * 60)
