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