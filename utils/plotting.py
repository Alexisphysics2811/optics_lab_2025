"""
Plotting utilities and style configurations for optics experiments.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_scientific_style():
    """
    Set matplotlib style for scientific plots.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set default parameters
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 8


def save_figure(filename, dpi=300, bbox_inches='tight', **kwargs):
    """
    Save the current figure with consistent settings.
    
    Parameters:
    -----------
    filename : str
        Output filename
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box setting
    **kwargs : dict
        Additional arguments passed to plt.savefig()
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved to: {filename}")


def create_subplot_grid(nrows, ncols, figsize=None):
    """
    Create a grid of subplots with consistent styling.
    
    Parameters:
    -----------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        Array of axes objects
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    plt.tight_layout()
    
    return fig, axes
