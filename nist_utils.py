import numpy as np
import re

def ExtractIntensityNist(column, default_value=1):
    """
    Extract numerical relative intensities from a NIST 'Rel.' column.

    Parameters
    ----------
    column : iterable
        Column containing NIST relative intensity values
        such as:
            '1000'
            '2qs'
            '50 w'
            '1000h'

    default_value : float
        Value assigned to invalid or missing entries.

    Returns
    -------
    intensity : numpy.ndarray
        Array of float intensities.
    """

    intensity = np.array([
        float(re.search(r'\d+\.?\d*', str(x)).group())
        if re.search(r'\d+\.?\d*', str(x))
        else 1
        for x in column
    ])

    return intensity


def Extract_Ei_Ek(column, default_value=np.nan):

    energy = np.array(column, dtype=str)

    Ei = []
    Ek = []

    for x in energy:

        # Case where entry is just "0"
        if x.strip() == "0":
            Ei.append(0.0)
            Ek.append(0.0)

        # Normal case: "Ei - Ek"
        elif "-" in x:
            left, right = x.split("-")

            try:
                Ei.append(float(left))
                Ek.append(float(right))
            except ValueError:
                Ei.append(default_value)
                Ek.append(default_value)

        # Fallback if format is unexpected
        else:
            Ei.append(default_value)
            Ek.append(default_value)

    Ei = np.array(Ei)
    Ek = np.array(Ek)

    return Ei, Ek


def Get_bremsstralung(lambda_nm , T , default_value=np.nan):

    # Physical constants
    h = 6.62607015e-34       # J.s
    c = 2.99792458e8        # m/s
    kB = 1.380649e-23       # J/K

    # Plasma parameters
    #T = 12000               # electron temperature [K]
    Z = 1                   # Ar+ approximation
    ne = 1e20               # electron density [m^-3]
    ni = ne / Z             # ion density [m^-3]
    gff = 1.0               # Gaunt factor ~ 1

    # Wavelength range nm->m
    lambda_m = lambda_nm * 1e-9               # m

    # Frequency
    nu = c / lambda_m

    # Bremsstrahlung spectral emissivity per Hz
    # Approximate SI scaling, arbitrary normalization acceptable for spectrum shape
    epsilon_nu = (
        Z**2 * ne * ni
        * T**(-0.5)
        * np.exp(-h * nu / (kB * T))
        * gff
    )

    # Convert from per Hz to per wavelength
    # epsilon_lambda = epsilon_nu * |dnu/dlambda| = epsilon_nu * c/lambda^2
    epsilon_lambda = epsilon_nu * c / lambda_m**2

    # Normalize for easier plotting
    epsilon_lambda_norm = epsilon_lambda / np.max(epsilon_lambda)

    return epsilon_lambda_norm