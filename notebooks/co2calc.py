import numpy as np

T0_Kelvin = 273.15
rho_ref = 1026.0

pH0 = 8.0
co2_chem_tol = 1.0e-12
co2_ph_high = 10.0
co2_ph_low = 6.0


def _mmolm3_to_molkg(value):
    return value * 1.0e-3 / rho_ref


def co2_eq_const(S, T):

    t_kel = T + T0_Kelvin
    t_sca = t_kel * 0.01
    t_sq = t_sca * t_sca
    t_inv = 1.0 / t_kel
    t_log = np.log(t_kel)

    s_sq = S * S
    s_sqrt = np.sqrt(S)
    s_1p5 = S ** 1.5
    s_cl = S / 1.80655

    s_sc = 19.924 * S / (1000.0 - 1.005 * S)
    s_sc_sq = s_sc * s_sc
    s_sc_sqrt = np.sqrt(s_sc)

    eq = {}
    # Mehrbach on pH_total
    eq["k_h2co3"] = 10.0 ** (
        -3633.86 / t_kel + 61.2172 - 9.6777 * t_log + 0.011555 * S - 0.0001152 * s_sq
    )

    eq["k_hco3"] = 10.0 ** (
        -471.78 / t_kel - 25.9290 + 3.16967 * t_log + 0.01781 * S - 0.0001122 * s_sq
    )

    eq["k_h3po4"] = np.exp(
        -4576.752 * t_inv
        + 115.540
        - 18.453 * t_log
        + (-106.736 * t_inv + 0.69171) * s_sqrt
        + (-0.65643 * t_inv - 0.01844) * S
    )

    eq["k_h2po4"] = np.exp(
        -8814.715 * t_inv
        + 172.0883
        - 27.927 * t_log
        + (-160.340 * t_inv + 1.3566) * s_sqrt
        + (0.37335 * t_inv - 0.05778) * S
    )

    eq["k_hpo4"] = np.exp(
        -3070.75 * t_inv
        - 18.126
        + (17.27039 * t_inv + 2.81197) * s_sqrt
        + (-44.99486 * t_inv - 0.09984) * S
    )

    eq["k_sioh4"] = np.exp(
        -8904.2 * t_inv
        + 117.385
        - 19.334 * t_log
        + (-458.79 * t_inv + 3.5913) * s_sc_sqrt
        + (188.74 * t_inv - 1.5998) * s_sc
        + (-12.1652 * t_inv + 0.07871) * s_sc_sq
        + np.log(1.0 - 0.001005 * S)
    )

    # following Zeebe and Wolf-Gladrow
    eq["k_oh"] = np.exp(
        -13847.26 * t_inv
        + 148.96502
        - 23.6521 * t_log
        + (118.67 * t_inv - 5.977 + 1.0495 * t_log) * s_sqrt
        - 0.01615 * S
    )

    eq["k_hso4"] = np.exp(
        -4276.1 * t_inv
        + 141.328
        - 23.093 * t_log
        + (-13856.0 * t_inv + 324.57 - 47.986 * t_log) * s_sc_sqrt
        + (35474 * t_inv - 771.54 + 114.723 * t_log) * s_sc
        - 2698 * t_inv * s_sc ** 1.5
        + 1776 * t_inv * s_sc_sq
        + np.log(1.0 - 0.001005 * S)
    )

    eq["boron_total"] = 0.000232 * s_cl / 10.811
    eq["sulfate"] = 0.14 * s_cl / 96.062
    eq["fluoride"] = 0.000067 * s_cl / 18.9984

    eq["k_hf"] = np.exp(
        1590.2 * t_inv
        - 12.641
        + 1.525 * s_sc_sqrt
        + np.log(1.0 - 0.001005 * S)
        + np.log(1.0 + (eq["sulfate"] / eq["k_hso4"]))
    )

    eq["k_hbo2"] = np.exp(
        (-8966.90 - 2890.53 * s_sqrt - 77.942 * S + 1.728 * s_1p5 - 0.0996 * s_sq)
        * t_inv
        + (148.0248 + 137.1942 * s_sqrt + 1.62142 * S)
        + (-24.4344 - 25.085 * s_sqrt - 0.2474 * S) * t_log
        + 0.053105 * s_sqrt * t_kel
    )
    return eq


def newton_safe(funcd, x_guess, x1, x2, xacc, **kwargs):
    MAXIT = 100
    fl, df = funcd(x1, **kwargs)
    fh, df = funcd(x2, **kwargs)

    xl = np.where(fl < 0, x1, x2)
    xh = np.where(fl < 0, x2, x1)

    rts = x_guess
    dxold = abs(x2 - x1)
    dx = dxold
    f, df = funcd(rts, **kwargs)

    for j in range(0, MAXIT):
        if ((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0 or abs(2.0 * f) > abs(
            dxold * df
        ):
            dxold = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dxold = dx
            dx = f / df
            temp = rts
            rts = rts - dx
            if temp == rts:
                return rts
        if abs(dx) < xacc:
            return rts

        f, df = funcd(rts, **kwargs)

        if f < 0:
            xl = rts
        else:
            xh = rts

    return rts


def calc_pH_from_alk_pco2(
    h,
    alk,
    pco2,
    boron_total,
    fluoride,
    phosphate,
    silicate,
    sulfate,
    k0,
    k_h2co3,
    k_hco3,
    k_hbo2,
    k_h3po4,
    k_h2po4,
    k_hpo4,
    k_hf,
    k_hso4,
    k_oh,
    k_sioh4,
):

    h_2 = h * h
    h_3 = h_2 * h
    k_01 = k0 * k_h2co3
    k_012 = k_01 * k_hco3
    k_12 = k_h2co3 * k_hco3
    k_12p = k_h3po4 * k_h2po4
    k_123p = k_12p * k_hpo4
    c = 1.0 + sulfate / k_hso4 + fluoride / k_hf
    a = h_3 + k_h3po4 * h_2 + k_12p * h + k_123p
    a2 = a * a
    da = 3.0 * h_2 + 2.0 * k_h3po4 * h + k_12p
    b = h_2 + k_h2co3 * h + k_12

    # Calculate F:
    # F = HCO3 + CO3 + Borate + OH + HPO4 + 2 * PO4 + Silicate + HFREE
    #       + HSO4 + HF + H3PO4 - TA
    f = (
        k_01 * pco2 / h
        + 2 * k_012 * pco2 / h_2
        + boron_total / (1.0 + h / k_hbo2)
        + k_oh / h
        + (k_12p * h + 2.0 * k_123p - h_3) * phosphate / a
        + silicate / (1.0 + h / k_sioh4)
        - h / c
        - sulfate / (1.0 + k_hso4 * c / h)
        - fluoride / (1.0 + k_hf * c / h)
        - alk
    )

    # calculate df=df/dh

    df = (
        -k_01 * pco2 / h_2
        - 4 * k_012 * pco2 / h_3
        - boron_total / (k_hbo2 * (1.0 + h / k_hbo2) ** 2)
        - k_oh / h_2
        + (k_12p * (a - h * da) - 2.0 * k_123p * da - h_2 * (3.0 * a - h * da))
        * phosphate
        / a2
        - silicate / (k_sioh4 * (1.0 + h / k_sioh4) ** 2)
        - 1.0 / c
        - sulfate / (1.0 + k_hso4 * c / h) ** 2.0 * (k_hso4 * c / h_2)
        - fluoride / (1.0 + k_hf * c / h) ** 2.0 * (k_hf * c / h_2)
    )

    return f, df


def _calc_pH_from_dic_alk(
    x,
    dic,
    alk,    
    boron_total,
    fluoride,
    phosphate,
    silicate,
    sulfate,
    k_h2co3,
    k_hco3,
    k_hbo2,
    k_h3po4,
    k_h2po4,
    k_hpo4,
    k_hf,
    k_hso4,
    k_oh,
    k_sioh4,
):

    x_2 = x * x
    x_3 = x_2 * x
    k_12 = k_h2co3 * k_hco3
    k_12p = k_h3po4 * k_h2po4
    k_123p = k_12p * k_hpo4
    c = 1.0 + sulfate / k_hso4 + fluoride / k_hf
    a = x_3 + k_h3po4 * x_2 + k_12p * x + k_123p
    a2 = a * a
    da = 3.0 * x_2 + 2.0 * k_h3po4 * x + k_12p
    b = x_2 + k_h2co3 * x + k_12
    b2 = b * b
    db = 2.0 * x + k_h2co3

    # Calculate F:
    # F = HCO3 + CO3 + Borate + OH + HPO4 + 2 * PO4 + Silicate + HFREE  \
    #       + HSO4 + HF + H3PO4 - TA
    f = (
        (k_h2co3 * x + 2.0 * k_12) * dic / b
        + boron_total / (1.0 + x / k_hbo2)
        + k_oh / x
        + (k_12p * x + 2.0 * k_123p - x_3) * phosphate / a
        + silicate / (1.0 + x / k_sioh4)
        - x / c
        - sulfate / (1.0 + k_hso4 * c / x)
        - fluoride / (1.0 + k_hf * c / x)
        - alk
    )

    # calculate df=df/dx

    df = (
        ((b - x * db) * k_h2co3 - 2.0 * k_12 * db) * dic / b2
        - boron_total / (k_hbo2 * (1.0 + x / k_hbo2) ** 2)
        - k_oh / x_2
        + (k_12p * (a - x * da) - 2.0 * k_123p * da - x_2 * (3.0 * a - x * da))
        * phosphate
        / a2
        - silicate / (k_sioh4 * (1.0 + x / k_sioh4) ** 2)
        - 1.0 / c
        - sulfate / (1.0 + k_hso4 * c / x) ** 2.0 * (k_hso4 * c / x_2)
        - fluoride / (1.0 + k_hf * c / x) ** 2.0 * (k_hf * c / x_2)
    )

    return f, df


def calc_pH_from_dic_alk(
    DIC,
    ALK,
    S,
    T,
    PO4=0.5,
    SiO3=10.0,
    input_in_gravimetric_units=False,
    pH0=8.0,
):

    thermodyn = co2_eq_const(S, T)

    if not input_in_gravimetric_units:
        dic_loc = _mmolm3_to_molkg(DIC)
        alk_loc = _mmolm3_to_molkg(ALK)
        phosphate_loc = _mmolm3_to_molkg(PO4)
        silicate_loc = _mmolm3_to_molkg(SiO3)
    else:
        # assume units are µmol/kg, covert to mol/kg
        dic_loc = 1e-6 * DIC
        alk_loc = 1e-6 * ALK
        phosphate_loc = 1e-6 * PO4
        silicate_loc = 1e-6 * SiO3

    h_total = np.vectorize(newton_safe)(
        _calc_pH_from_dic_alk,
        10.0 ** (-pH0),
        10.0 ** (-co2_ph_low),
        10.0 ** (-co2_ph_high),
        co2_chem_tol,
        dic=dic_loc,
        alk=alk_loc,
        phosphate=phosphate_loc,
        silicate=silicate_loc,
        **thermodyn
    )
    return -1.0 * np.log10(h_total)


def calc_co2(
    DIC,
    ALK,
    S,
    T,
    PO4=0.5,
    SiO3=10.0,
    input_in_gravimetric_units=False,
    pH0=8.0,
):
    """compute CO2aq from DIC and ALk"""
    thermodyn = co2_eq_const(S, T)

    pH = calc_pH_from_dic_alk(
        DIC,
        ALK,
        S,
        T,
        PO4,
        SiO3,
        input_in_gravimetric_units,
        pH0,
    )

    if not input_in_gravimetric_units:
        dic_loc = _mmolm3_to_molkg(DIC)
    else:
        # assume units are µmol/kg, covert to mol/kg
        dic_loc = 1e-6 * DIC

    h_total = 10.0 ** (-1.0 * pH)
    h2 = h_total * h_total

    co2aq = (
        dic_loc
        * h2
        / (
            h2
            + thermodyn["k_h2co3"] * h_total
            + thermodyn["k_h2co3"] * thermodyn["k_hco3"]
        )
    )
    if not input_in_gravimetric_units:
        return co2aq * 1.0e3 * rho_ref  # covert to mmol/m^3
    else:
        return co2aq * 1.0e6  # µmol/kg


def calc_dic(
    ALK,
    pCO2,
    S,
    T,    
    PO4=0.5,
    SiO3=10.0,
    input_in_gravimetric_units=False,
    pH0=8.0,
):

    thermodyn = co2_eq_const(S, T)
    thermodyn["k0"] = 1e-6 * co2sol(S, T, return_in_gravimetric_units=True)

    if not input_in_gravimetric_units:
        alk_loc = _mmolm3_to_molkg(ALK)
        phosphate_loc = _mmolm3_to_molkg(PO4)
        silicate_loc = _mmolm3_to_molkg(SiO3)
    else:
        # assume units are µmol/kg, covert to mol/kg
        alk_loc = 1e-6 * ALK
        phosphate_loc = 1e-6 * PO4
        silicate_loc = 1e-6 * SiO3

    pco2_loc = pCO2 * 1e-6

    h_total = np.vectorize(newton_safe)(
        calc_pH_from_alk_pco2,
        10.0 ** (-pH0),
        10.0 ** (-co2_ph_low),
        10.0 ** (-co2_ph_high),
        co2_chem_tol,
        alk=alk_loc,
        pco2=pco2_loc,
        phosphate=phosphate_loc,
        silicate=silicate_loc,
        **thermodyn
    )

    # Solve carbonate chemistry in surface
    h2 = h_total * h_total
    co2aq = thermodyn["k0"] * pco2_loc  # mol kg^{-1}

    dic = co2aq * (
        1.0
        + thermodyn["k_h2co3"] / h_total
        + thermodyn["k_h2co3"] * thermodyn["k_hco3"] / h2
    )

    if not input_in_gravimetric_units:
        return dic * 1.0e3 * rho_ref  # covert to mmol/m^3
    else:
        return dic * 1.0e6  # µmol/kg


def co2sol(S, T, return_in_gravimetric_units=False):
    """
    Solubility of CO2 in sea water
    INPUT:
    S = salinity    [PSS]
    T = temperature [degree C]

    conc = solubility of CO2 [mmol/m^3/ppm]
    Weiss & Price (1980, Mar. Chem., 8, 347-359;
    Eq 13 with table 6 values)
    """

    a = np.array([-162.8301, 218.2968, 90.9241, -1.47696])
    b = np.array([0.025695, -0.025225, 0.0049867])

    T_sc = (T + T0_Kelvin) * 0.01
    T_sq = T_sc * T_sc
    T_inv = 1.0 / T_sc
    log_T = np.log(T_sc)
    d0 = b[2] * T_sq + b[1] * T_sc + b[0]

    # compute CO2 solubility in mol.kg^{-1}.atm^{-1}
    co2_sol = np.exp(a[0] + a[1] * T_inv + a[2] * log_T + a[3] * T_sq + d0 * S)

    if return_in_gravimetric_units:
        return 1.0e6 * co2_sol  # µmol/kg/atm
    else:
        # convert to mmol/m^3/muatm
        return co2_sol * rho_ref * 1.0e3
