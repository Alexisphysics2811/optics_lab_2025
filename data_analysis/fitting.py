"""
Common curve fitting functions for optics experiments.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


def linear_fit(x, y, return_stats=True):
    """
    Perform linear regression fit.
    
    Parameters:
    -----------
    x : array_like
        Independent variable data
    y : array_like
        Dependent variable data
    return_stats : bool
        Whether to return fit statistics
        
    Returns:
    --------
    slope : float
        Slope of the line
    intercept : float
        Y-intercept
    stats_dict : dict (if return_stats=True)
        Dictionary with fit statistics (r_value, p_value, std_err)
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    if return_stats:
        stats_dict = {
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
        return slope, intercept, stats_dict
    else:
        return slope, intercept


def gaussian(x, amplitude, center, width, offset=0):
    """
    Gaussian function.
    
    Parameters:
    -----------
    x : array_like
        Input values
    amplitude : float
        Peak amplitude
    center : float
        Center position (mean)
    width : float
        Standard deviation (sigma)
    offset : float
        Vertical offset
        
    Returns:
    --------
    y : array_like
        Gaussian values
    """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2)) + offset


def fit_gaussian(x, y, initial_guess=None):
    """
    Fit data to a Gaussian function.
    
    Parameters:
    -----------
    x : array_like
        Independent variable data
    y : array_like
        Dependent variable data
    initial_guess : tuple, optional
        Initial guess for parameters (amplitude, center, width, offset)
        
    Returns:
    --------
    params : array
        Fitted parameters [amplitude, center, width, offset]
    covariance : array
        Covariance matrix of parameters
    """
    if initial_guess is None:
        # Estimate initial parameters
        amplitude_guess = np.max(y) - np.min(y)
        center_guess = x[np.argmax(y)]
        width_guess = (np.max(x) - np.min(x)) / 10
        offset_guess = np.min(y)
        initial_guess = [amplitude_guess, center_guess, width_guess, offset_guess]
    
    params, covariance = curve_fit(gaussian, x, y, p0=initial_guess)
    return params, covariance


def sinusoidal(x, amplitude, frequency, phase, offset):
    """
    Sinusoidal function.
    
    Parameters:
    -----------
    x : array_like
        Input values
    amplitude : float
        Amplitude
    frequency : float
        Angular frequency (rad/unit)
    phase : float
        Phase shift (radians)
    offset : float
        Vertical offset
        
    Returns:
    --------
    y : array_like
        Sinusoidal values
    """
    return amplitude * np.sin(frequency * x + phase) + offset


def fit_sinusoid(x, y, initial_guess=None):
    """
    Fit data to a sinusoidal function.
    
    Parameters:
    -----------
    x : array_like
        Independent variable data
    y : array_like
        Dependent variable data
    initial_guess : tuple, optional
        Initial guess for parameters (amplitude, frequency, phase, offset)
        
    Returns:
    --------
    params : array
        Fitted parameters [amplitude, frequency, phase, offset]
    covariance : array
        Covariance matrix of parameters
    """
    if initial_guess is None:
        # Estimate initial parameters
        amplitude_guess = (np.max(y) - np.min(y)) / 2
        offset_guess = np.mean(y)
        
        # Estimate frequency using FFT
        fft = np.fft.fft(y - offset_guess)
        frequencies = np.fft.fftfreq(len(x), x[1] - x[0])
        frequency_guess = 2 * np.pi * np.abs(frequencies[np.argmax(np.abs(fft[1:])) + 1])
        
        phase_guess = 0
        initial_guess = [amplitude_guess, frequency_guess, phase_guess, offset_guess]
    
    params, covariance = curve_fit(sinusoidal, x, y, p0=initial_guess)
    return params, covariance
