"""
Brain Colours - Perceptually Uniform Colour Maps for Brain Imaging

This module provides perceptually uniform colour maps and utilities for brain imaging.
Based on Peter Kovesi's work: https://arxiv.org/abs/1509.03700

Reference:
    "The CIEDE2000 Color-Difference Formula: Implementation Notes,
    Supplementary Test Data, and Mathematical Observations,"
    G. Sharma, W. Wu, E. N. Dalal,
    Color Research and Application, vol. 30. No. 1, pp. 21-30, February 2005.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import warnings


# ============================================================================
# Helper Functions for make_newmap
# ============================================================================

def ch2ab(chroma: float, angle_degrees: float) -> tuple:
    """
    Convert from (chroma, hue angle) to (a*, b*) coordinates in CIE L*a*b*.
    
    Parameters
    ----------
    chroma : float
        Chroma value
    angle_degrees : float
        Hue angle in degrees
        
    Returns
    -------
    tuple
        (a*, b*) coordinates as tuple for unpacking
    """
    theta = np.deg2rad(angle_degrees)
    a_star = chroma * np.cos(theta)
    b_star = chroma * np.sin(theta)
    return (a_star, b_star)


def bbspline(control_points: np.ndarray, spline_order: int, n_samples: int) -> np.ndarray:
    """
    Basic B-spline interpolation through control points.
    
    Parameters
    ----------
    control_points : np.ndarray
        Control points, shape (n_dims, n_points)
    spline_order : int
        Order of the spline (2=linear, 3=quadratic, etc.)
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    np.ndarray
        Interpolated values, shape (n_dims, n_samples)
    """
    n_dims, n_pts = control_points.shape
    
    if n_pts < 2:
        raise ValueError("Need at least 2 control points")
    
    if spline_order > n_pts:
        spline_order = n_pts
        warnings.warn(f"Spline order reduced to {n_pts}")
    
    # Parameter values for control points
    t = np.linspace(0, 1, n_pts)
    
    # Parameter values for sampling
    t_new = np.linspace(0, 1, n_samples)
    
    # Interpolate each dimension
    result = np.zeros((n_dims, n_samples))
    k = min(spline_order, n_pts - 1)  # scipy uses k+1 order
    
    for dim in range(n_dims):
        # Use B-spline with clamped boundary conditions
        tck = interpolate.splrep(t, control_points[dim, :], k=k, s=0)
        result[dim, :] = interpolate.splev(t_new, tck)
    
    return result


def nc_colour_maps(N_colours: int) -> np.ndarray:
    """
    Return non-continuous colour maps as RGB triplets.
    
    For N_colours <= 11, uses ColorBrewer qualitative palette with colour-blind safe options.
    For N_colours > 11, samples at equidistant points from the xrain colour scale.
    
    Parameters
    ----------
    N_colours : int
        Number of distinct colours needed
        
    Returns
    -------
    np.ndarray
        RGB colour map, shape (N_colours, 3), values in [0, 1]
        
    References
    ----------
    ColorBrewer: http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
    xrain scale: https://github.com/CPernet/brain_colours
    """
    if N_colours <= 11:
        # ColorBrewer qualitative palette (colour-blind safe)
        colorbrewer = np.array([
            [31, 120, 180],
            [178, 223, 138],
            [51, 160, 44],
            [251, 154, 153],
            [227, 26, 28],
            [253, 191, 111],
            [255, 127, 0],
            [202, 178, 214],
            [106, 61, 154],
            [255, 255, 153],
            [177, 89, 40]
        ], dtype=float) / 255.0
        
        return colorbrewer[:N_colours, :]
    
    else:
        # xrain colour scale (256 entries)
        xrain = np.array([
            [0, 0.0045, 0], [0.0167, 0.0055, 0.0044], [0.0377, 0.0066, 0.0424],
            [0.0563, 0.0079, 0.0715], [0.0711, 0.0093, 0.0927], [0.0832, 0.0110, 0.1107],
            [0.0924, 0.0133, 0.1282], [0.0990, 0.0159, 0.1456], [0.1040, 0.0188, 0.1628],
            [0.1089, 0.0210, 0.1797], [0.1139, 0.0225, 0.1963], [0.1192, 0.0233, 0.2126],
            [0.1245, 0.0233, 0.2286], [0.1301, 0.0227, 0.2443], [0.1357, 0.0213, 0.2598],
            [0.1414, 0.0194, 0.2749], [0.1470, 0.0175, 0.2898], [0.1524, 0.0157, 0.3045],
            [0.1577, 0.0139, 0.3190], [0.1629, 0.0123, 0.3334], [0.1678, 0.0104, 0.3478],
            [0.1725, 0.0090, 0.3621], [0.1770, 0.0076, 0.3765], [0.1812, 0.0063, 0.3909],
            [0.1852, 0.0052, 0.4055], [0.1889, 0.0041, 0.4202], [0.1922, 0.0032, 0.4352],
            [0.1951, 0.0024, 0.4504], [0.1977, 0.0017, 0.4658], [0.1997, 0.0012, 0.4815],
            [0.2014, 0.0009, 0.4974], [0.2025, 0.0007, 0.5136], [0.2032, 0.0007, 0.5300],
            [0.2033, 0.0009, 0.5466], [0.2029, 0.0015, 0.5633], [0.2019, 0.0023, 0.5802],
            [0.2003, 0.0036, 0.5970], [0.1982, 0.0054, 0.6138], [0.1955, 0.0077, 0.6305],
            [0.1922, 0.0107, 0.6470], [0.1885, 0.0148, 0.6631], [0.1842, 0.0197, 0.6788],
            [0.1794, 0.0257, 0.6939], [0.1743, 0.0330, 0.7083], [0.1689, 0.0420, 0.7220],
            [0.1632, 0.0515, 0.7348], [0.1573, 0.0614, 0.7465], [0.1514, 0.0718, 0.7572],
            [0.1456, 0.0824, 0.7666], [0.1398, 0.0933, 0.7748], [0.1345, 0.1045, 0.7816],
            [0.1294, 0.1160, 0.7871], [0.1249, 0.1276, 0.7911], [0.1209, 0.1394, 0.7937],
            [0.1176, 0.1514, 0.7949], [0.1149, 0.1634, 0.7947], [0.1129, 0.1754, 0.7931],
            [0.1115, 0.1874, 0.7902], [0.1106, 0.1993, 0.7861], [0.1102, 0.2113, 0.7809],
            [0.1101, 0.2230, 0.7747], [0.1102, 0.2347, 0.7676], [0.1105, 0.2462, 0.7597],
            [0.1109, 0.2575, 0.7511], [0.1112, 0.2686, 0.7419], [0.1115, 0.2796, 0.7323],
            [0.1116, 0.2903, 0.7223], [0.1116, 0.3008, 0.7121], [0.1114, 0.3111, 0.7017],
            [0.1110, 0.3212, 0.6913], [0.1104, 0.3311, 0.6808], [0.1096, 0.3408, 0.6704],
            [0.1086, 0.3503, 0.6601], [0.1075, 0.3596, 0.6499], [0.1063, 0.3688, 0.6399],
            [0.1049, 0.3778, 0.6300], [0.1035, 0.3866, 0.6204], [0.1019, 0.3953, 0.6109],
            [0.1003, 0.4039, 0.6017], [0.0987, 0.4123, 0.5927], [0.0971, 0.4206, 0.5838],
            [0.0955, 0.4289, 0.5751], [0.0939, 0.4370, 0.5666], [0.0923, 0.4450, 0.5582],
            [0.0907, 0.4529, 0.5500], [0.0891, 0.4608, 0.5420], [0.0876, 0.4686, 0.5340],
            [0.0859, 0.4763, 0.5262], [0.0845, 0.4839, 0.5184], [0.0829, 0.4915, 0.5108],
            [0.0815, 0.4991, 0.5032], [0.0800, 0.5065, 0.4958], [0.0786, 0.5140, 0.4884],
            [0.0772, 0.5214, 0.4811], [0.0759, 0.5288, 0.4738], [0.0745, 0.5361, 0.4666],
            [0.0731, 0.5434, 0.4594], [0.0717, 0.5507, 0.4523], [0.0702, 0.5579, 0.4452],
            [0.0689, 0.5651, 0.4381], [0.0673, 0.5723, 0.4311], [0.0658, 0.5795, 0.4241],
            [0.0643, 0.5866, 0.4172], [0.0629, 0.5938, 0.4103], [0.0612, 0.6009, 0.4034],
            [0.0598, 0.6080, 0.3965], [0.0581, 0.6151, 0.3896], [0.0566, 0.6222, 0.3828],
            [0.0549, 0.6292, 0.3760], [0.0532, 0.6363, 0.3692], [0.0516, 0.6433, 0.3625],
            [0.0499, 0.6503, 0.3557], [0.0482, 0.6574, 0.3490], [0.0465, 0.6644, 0.3423],
            [0.0448, 0.6714, 0.3356], [0.0429, 0.6784, 0.3290], [0.0413, 0.6854, 0.3223],
            [0.0395, 0.6924, 0.3157], [0.0377, 0.6993, 0.3091], [0.0360, 0.7063, 0.3024],
            [0.0341, 0.7133, 0.2959], [0.0324, 0.7203, 0.2893], [0.0308, 0.7273, 0.2828],
            [0.0292, 0.7342, 0.2762], [0.0276, 0.7412, 0.2697], [0.0261, 0.7451, 0.2632],
            [0.0246, 0.7551, 0.2567], [0.0232, 0.7621, 0.2502], [0.0218, 0.7691, 0.2437],
            [0.0205, 0.7760, 0.2373], [0.0193, 0.7830, 0.2308], [0.0183, 0.7900, 0.2243],
            [0.0176, 0.7969, 0.2179], [0.0173, 0.8039, 0.2115], [0.0176, 0.8109, 0.2051],
            [0.0186, 0.8178, 0.1987], [0.0204, 0.8247, 0.1924], [0.0233, 0.8317, 0.1860],
            [0.0276, 0.8386, 0.1796], [0.0336, 0.8455, 0.1733], [0.0421, 0.8524, 0.1670],
            [0.0518, 0.8592, 0.1607], [0.0630, 0.8661, 0.1544], [0.0753, 0.8729, 0.1481],
            [0.0890, 0.8796, 0.1418], [0.1037, 0.8863, 0.1356], [0.1197, 0.8929, 0.1293],
            [0.1368, 0.8994, 0.1231], [0.1550, 0.9058, 0.1169], [0.1742, 0.9121, 0.1107],
            [0.1945, 0.9183, 0.1044], [0.2157, 0.9244, 0.0983], [0.2378, 0.9303, 0.0921],
            [0.2609, 0.9360, 0.0859], [0.2849, 0.9415, 0.0797], [0.3096, 0.9467, 0.0736],
            [0.3351, 0.9517, 0.0674], [0.3612, 0.9565, 0.0612], [0.3879, 0.9609, 0.0551],
            [0.4152, 0.9651, 0.0490], [0.4428, 0.9689, 0.0427], [0.4707, 0.9723, 0.0367],
            [0.4987, 0.9754, 0.0309], [0.5269, 0.9780, 0.0258], [0.5550, 0.9802, 0.0212],
            [0.5829, 0.9820, 0.0171], [0.6106, 0.9833, 0.0135], [0.6378, 0.9841, 0.0101],
            [0.6645, 0.9845, 0.0073], [0.6906, 0.9843, 0.0050], [0.7159, 0.9836, 0.0030],
            [0.7403, 0.9825, 0.0013], [0.7638, 0.9808, 0], [0.7862, 0.9786, 0],
            [0.8075, 0.9759, 0], [0.8275, 0.9728, 0], [0.8464, 0.9691, 0],
            [0.8640, 0.9650, 0], [0.8803, 0.9605, 0], [0.8953, 0.9556, 0],
            [0.9090, 0.9502, 0], [0.9214, 0.9445, 0], [0.9327, 0.9385, 0],
            [0.9427, 0.9321, 0], [0.9517, 0.9255, 0], [0.9596, 0.9185, 0],
            [0.9665, 0.9113, 0], [0.9725, 0.9039, 0], [0.9777, 0.8963, 0],
            [0.9821, 0.8885, 0], [0.9859, 0.8806, 0], [0.9891, 0.8725, 0],
            [0.9917, 0.8642, 0], [0.9939, 0.8559, 0], [0.9957, 0.8475, 0],
            [0.9972, 0.8390, 0], [0.9984, 0.8304, 0], [0.9993, 0.8217, 0],
            [1.0000, 0.8130, 0], [1.0000, 0.8043, 0], [1.0000, 0.7954, 0],
            [1.0000, 0.7866, 0], [1.0000, 0.7776, 0], [1.0000, 0.7687, 0],
            [1.0000, 0.7597, 0], [1.0000, 0.7506, 0], [1.0000, 0.7415, 0],
            [1.0000, 0.7324, 0], [1.0000, 0.7232, 0], [1.0000, 0.7140, 0],
            [1.0000, 0.7047, 0], [1.0000, 0.6954, 0], [1.0000, 0.6860, 0],
            [1.0000, 0.6766, 0], [1.0000, 0.6671, 0], [1.0000, 0.6576, 0],
            [1.0000, 0.6479, 0], [1.0000, 0.6383, 0], [1.0000, 0.6285, 0],
            [1.0000, 0.6187, 0], [1.0000, 0.6088, 0], [1.0000, 0.5988, 0],
            [1.0000, 0.5887, 0], [1.0000, 0.5786, 0], [1.0000, 0.5683, 0],
            [1.0000, 0.5579, 0], [1.0000, 0.5474, 0], [1.0000, 0.5368, 0],
            [1.0000, 0.5261, 0], [1.0000, 0.5152, 0], [1.0000, 0.5042, 0],
            [1.0000, 0.4930, 0], [1.0000, 0.4816, 0], [1.0000, 0.4701, 0],
            [1.0000, 0.4583, 0], [1.0000, 0.4464, 0], [1.0000, 0.4342, 0],
            [1.0000, 0.4217, 0], [1.0000, 0.4089, 0], [1.0000, 0.3959, 0],
            [1.0000, 0.3824, 0], [1.0000, 0.3686, 0], [1.0000, 0.3544, 0],
            [1.0000, 0.3396, 0], [1.0000, 0.3243, 0], [1.0000, 0.3083, 0],
            [1.0000, 0.2915, 0], [1.0000, 0.2738, 0], [1.0000, 0.2549, 0],
            [1.0000, 0.2347, 0], [1.0000, 0.2126, 0], [1.0000, 0.1881, 0],
            [1.0000, 0.1600, 0], [1.0000, 0.1263, 0], [1.0000, 0.0811, 0],
            [1.0000, 0.0026, 0]
        ])
        
        # Sample at equidistant points
        step = 256 // N_colours
        indices = np.arange(0, 256, step)[:N_colours]
        return xrain[indices, :]


# ============================================================================
# Main make_newmap Function (adapted from cmap.m)
# ============================================================================

def make_newmap(label: str, save_as: Optional[str] = None) -> Tuple[np.ndarray, str, str]:
    """
    Generate perceptually uniform colour maps.
    
    This is the Python equivalent of Peter Kovesi's cmap.m function.
    
    Parameters
    ----------
    label : str
        Colour map label. Options include:
        - Linear: 'grey', 'BRYW', 'BRY', 'blue' (L1, L3, L4, L6)
        - Diverging: 'BWR', 'GWR', 'BGY', 'BGR' (D1, D3, D7, D8)
        - Non-continuous: '[N]_colours' where N is an integer (e.g., '5_colours', '12_colours')
    save_as : str, optional
        File path to save the colormap. File extension determines format:
        - '.mat': MATLAB format
        - '.lut': ImageJ/MRIcron format (binary)
        - '.cmap': FSLeyes format (text)
        - '.txt' or '.csv': Plain text CSV format
        If provided, the colormap will be saved to disk.
        
    Returns
    -------
    colormap : np.ndarray
        RGB colour map, shape (256, 3), values in [0, 1]
    name : str
        Descriptive name of the colour map
    desc : str
        Brief description
        
    Examples
    --------
    >>> # Grey scale
    >>> cmap_grey, name, desc = make_newmap('grey')
    
    >>> # Heat map
    >>> cmap_heat, name, desc = make_newmap('BRYW')
    
    >>> # Diverging blue-white-red
    >>> cmap_bwr, name, desc = make_newmap('BWR')
    
    >>> # Save colormap to file
    >>> cmap, name, desc = make_newmap('BWR', save_as='blue-white-red.lut')
    
    >>> # Non-continuous colormap for discrete data
    >>> cmap_5, name, desc = make_newmap('5_colours')
    >>> cmap_12, name, desc = make_newmap('12_colours')
    
    References
    ----------
    Peter Kovesi. Good Colour Maps: How to Design Them.
    arXiv:1509.03700 [cs.GR] 2015
    https://arxiv.org/abs/1509.03700
    """
    # Default values
    N = 256
    chroma_k = 1.0
    reverse = False
    diagnostics = False
    
    label = label.strip()
    
    # Check for N_colours pattern (non-continuous maps)
    if '_colours' in label.lower() or '_colors' in label.lower():
        try:
            # Extract N from pattern like '5_colours' or '12_colors'
            n_str = label.lower().replace('_colours', '').replace('_colors', '').strip()
            n_colours = int(n_str)
            if n_colours < 1:
                raise ValueError("Number of colours must be >= 1")
            
            # Generate non-continuous colormap
            colormap = nc_colour_maps(n_colours)
            name = f"nc_{n_colours}_colours"
            desc = f"Non-continuous colour map with {n_colours} distinct colours."
            
            # Save if requested
            if save_as is not None:
                save_path = Path(save_as)
                extension = save_path.suffix.lower()
                
                if extension == '.mat':
                    try:
                        import scipy.io
                        scipy.io.savemat(save_path, {'colormap': colormap, 'name': name, 'description': desc})
                    except ImportError:
                        raise ImportError("scipy is required to save .mat files. Install with: pip install scipy")
                elif extension == '.lut':
                    lut_bytes = (colormap * 255).astype(np.uint8)
                    with open(save_path, 'wb') as f:
                        lut_bytes.tofile(f)
                elif extension == '.cmap':
                    np.savetxt(save_path, colormap, fmt='%.6f')
                elif extension in ['.txt', '.csv']:
                    np.savetxt(save_path, colormap, fmt='%.6f', delimiter=',')
                else:
                    raise ValueError(f"Unsupported file extension '{extension}'. "
                                   f"Supported formats: .mat, .lut, .cmap, .txt, .csv")
            
            return colormap, name, desc
            
        except ValueError as e:
            raise ValueError(f"Invalid N_colours format: '{label}'. "
                           f"Use format like '5_colours' or '12_colours'. Error: {e}")
    
    label = label.upper()
    
    # Default parameters
    colourspace = 'LAB'
    sigma = 0
    splineorder = 3
    formula = 'CIE76'
    W = np.array([1, 0, 0])
    desc = ''
    attribute_str = ''
    hue_str = ''
    
    # Colormap definitions
    colpts = None
    
    # ========================================================================
    # LINEAR COLOUR MAPS
    # ========================================================================
    
    if label.upper() in ['GREY', 'GRAY', 'L1']:
        desc = 'Standard grey scale.'
        attribute_str = 'linear'
        hue_str = 'grey'
        colpts = np.array([[0, 0, 0],
                          [100, 0, 0]], dtype=float).T
        splineorder = 2
        
    elif label.upper() in ['BRYW', 'L3']:
        desc = 'Black - Red - Yellow - White heat map.'
        attribute_str = 'linear'
        hue_str = 'bryw'
        colourspace = 'RGB'
        splineorder = 3
        colpts = np.array([[0, 0, 0],
                          [0.85, 0, 0],
                          [1, 0.15, 0],
                          [1, 0.85, 0],
                          [1, 1, 0.15],
                          [1, 1, 1]]).T
        
    elif label.upper() in ['BRY', 'L4']:
        desc = 'Black - Red - Yellow heat map.'
        attribute_str = 'linear'
        hue_str = 'bry'
        colourspace = 'RGB'
        splineorder = 3
        colpts = np.array([[0, 0, 0],
                          [0.85, 0, 0],
                          [1, 0.15, 0],
                          [1, 1, 0]]).T
        
    elif label.upper() in ['BLUE', 'L6']:
        desc = 'Blue shades running vertically up the edge of CIELAB space.'
        attribute_str = 'linear'
        hue_str = 'blue'
        colpts = np.array([[5, 31, -45],
                          [15, 50, -66],
                          [25, 65, -90],
                          [35, 70, -100],
                          [45, 45, -85],
                          [55, 20, -70],
                          [65, 0, -53],
                          [75, -22, -37],
                          [85, -38, -20],
                          [95, -25, -3]], dtype=float).T
    
    # ========================================================================
    # DIVERGING COLOUR MAPS
    # ========================================================================
    
    elif label.upper() in ['BWR', 'D1']:
        desc = 'Classic diverging Blue - White - Red colour map. End colours are matched in lightness and chroma.'
        attribute_str = 'diverging'
        hue_str = 'bwr'
        colpts = np.array([[40, *ch2ab(83, -64)],
                          [95, 0, 0],
                          [40, *ch2ab(83, 39)]]).T
        sigma = 7
        splineorder = 2
    
    elif label.upper() in ['GWR', 'D3']:
        desc = 'Diverging Green - White - Red colour map.'
        attribute_str = 'diverging'
        hue_str = 'gwr'
        colpts = np.array([[55, -50, 55],
                          [95, 0, 0],
                          [55, 63, 39]], dtype=float).T
        sigma = 7
        splineorder = 2
        
    elif label.upper() in ['BGY', 'D7']:
        desc = 'Linear-diverging Blue - Grey - Yellow colour map. This kind of diverging map has no perceptual dead spot at the centre.'
        attribute_str = 'diverging-linear'
        hue_str = 'bgy'
        colpts = np.array([[30, *ch2ab(89, -59)],
                          [60, 0, 0],
                          [90, *ch2ab(89, 96)]]).T
        splineorder = 2
        
    elif label.upper() in ['BGR', 'D8']:
        desc = 'Linear-diverging Blue - Grey - Red.'
        attribute_str = 'diverging-linear'
        hue_str = 'bgr'
        colpts = np.array([[30, *ch2ab(105, -58)],
                          [42.5, 0, 0],
                          [55, *ch2ab(105, 41)]]).T
        splineorder = 2
    
    else:
        raise ValueError(f"Unknown colour map label: '{label}'. "
                        f"Available options:\n"
                        f"  Linear: 'grey', 'BRYW', 'BRY', 'blue'\n"
                        f"  Diverging: 'BWR', 'GWR', 'BGY', 'BGR'")
    
    # ========================================================================
    # Generate colour map from control points
    # ========================================================================
    
    if colpts is None:
        raise ValueError(f"Colour map '{label}' not fully implemented yet")
    
    # Apply chroma scaling (only for LAB space)
    if colourspace.upper() == 'LAB' and colpts.shape[0] == 3:
        colpts[1:3, :] *= chroma_k
    
    # Generate spline path
    n_pts = colpts.shape[1]
    if n_pts < 2:
        raise ValueError("Need at least 2 control points")
    if n_pts < splineorder:
        splineorder = n_pts
        warnings.warn(f"Spline order reduced to {n_pts}")
    
    # Use basic B-spline (non-cyclic)
    labspline = bbspline(colpts, splineorder, N)
    
    # Clip RGB values if in RGB space (spline can overshoot)
    if colourspace.upper() == 'RGB':
        labspline = np.clip(labspline, 0, 1)
    
    # Apply contrast equalisation
    sigma_scaled = sigma * N / 256
    colormap = equalise_colourmap(colourspace, labspline.T, formula,
                                  W, sigma_scaled, diagnostics)
    
    # Reverse if requested
    if reverse:
        colormap = np.flipud(colormap)
    
    # Compute mean chroma
    lab = rgb_to_lab(colormap)
    mean_chroma = np.mean(np.sqrt(lab[:, 1]**2 + lab[:, 2]**2))
    
    # Construct lightness range description
    if colourspace.upper() == 'LAB':
        L_vals = colpts[0, :]
    else:
        L_vals = lab[:, 0]
    
    min_L = int(np.min(L_vals))
    max_L = int(np.max(L_vals))
    
    if min_L == max_L:
        L_str = f"{min_L}"
    else:
        L_str = f"{min_L}-{max_L}"
    
    # Build colour map name
    name = f"{attribute_str}_{hue_str}_{L_str}_c{int(round(mean_chroma))}_n{N}"
    
    if reverse:
        name += "_r"
    
    if diagnostics:
        print(f"Description: {desc}")
        print(f"Name: {name}")
        print(f"Lightness range: {L_str}")
        print(f"Mean chroma: {mean_chroma:.1f}")
    
    # Save colormap if requested
    if save_as is not None:
        save_path = Path(save_as)
        extension = save_path.suffix.lower()
        
        if extension == '.mat':
            # Save as MATLAB format
            try:
                import scipy.io
                scipy.io.savemat(save_path, {'colormap': colormap, 'name': name, 'description': desc})
            except ImportError:
                raise ImportError("scipy is required to save .mat files. Install with: pip install scipy")
        
        elif extension == '.lut':
            # Save as ImageJ/MRIcron format (binary)
            lut_bytes = (colormap * 255).astype(np.uint8)
            with open(save_path, 'wb') as f:
                lut_bytes.tofile(f)
        
        elif extension == '.cmap':
            # Save as FSLeyes format (text)
            np.savetxt(save_path, colormap, fmt='%.6f')
        
        elif extension in ['.txt', '.csv']:
            # Save as plain text CSV
            np.savetxt(save_path, colormap, fmt='%.6f', delimiter=',')
        
        else:
            raise ValueError(f"Unsupported file extension '{extension}'. "
                           f"Supported formats: .mat, .lut, .cmap, .txt, .csv")
    
    return colormap, name, desc


# ============================================================================
# Main map_luminance and map_isoluminance Functions
# ============================================================================

def map_luminance(path_to_maps: Union[str, Path], 
                  output_dir: Optional[Union[str, Path]] = None,
                  save_as: Optional[str] = None) -> None:
    """
    Fix luminance issues in colour maps.
    
    This function reads colour maps from a directory, applies luminance equalization
    (lightness correction only), and saves corrected versions.
    
    Parameters
    ----------
    path_to_maps : str or Path
        Path to directory containing colormap files (.mat, .lut, .clut, .cmap, .txt, .csv)
    output_dir : str or Path, optional
        Output directory. Defaults to 'new_braincolour_maps' in current directory
    save_as : str, optional
        Format to save: 'mat', 'lut', 'clut', 'cmap', 'txt', 'csv'
        If None, saves with same extension as input files (default: None)
        
    Notes
    -----
    Uses W=[1, 0, 0] for lightness-only correction.
    For full isoluminant correction, use map_isoluminance().
    """
    path_to_maps = Path(path_to_maps)
    
    if output_dir is None:
        output_dir = Path.cwd() / 'new_braincolour_maps'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all supported colormap files
    colormap_files = []
    for ext in ['*.lut', '*.clut', '*.cmap', '*.mat', '*.txt', '*.csv']:
        colormap_files.extend(list(path_to_maps.glob(ext)))
    
    if not colormap_files:
        print(f"No colormap files (.lut, .clut, .cmap, .mat, .txt, .csv) found in {path_to_maps}")
        return
    
    print(f"Processing {len(colormap_files)} colour maps...")
    
    for cmap_file in colormap_files:
        try:
            # Load the colour map based on file type
            base_name = cmap_file.stem
            ext = cmap_file.suffix.lower()
            
            if ext == '.mat':
                try:
                    import scipy.io
                    mat_data = scipy.io.loadmat(cmap_file)
                    # Try common variable names
                    if 'colormap' in mat_data:
                        lut_map = mat_data['colormap']
                    elif 'cmap' in mat_data:
                        lut_map = mat_data['cmap']
                    else:
                        # Get first non-metadata variable
                        keys = [k for k in mat_data.keys() if not k.startswith('__')]
                        if keys:
                            lut_map = mat_data[keys[0]]
                        else:
                            raise ValueError("No colormap data found in .mat file")
                except ImportError:
                    print(f"  ⚠ {cmap_file.name}: scipy required for .mat files")
                    continue
            elif ext in ['.txt', '.csv']:
                lut_map = np.loadtxt(cmap_file, delimiter=',')
            else:
                # .lut or .cmap
                lut_map = load_lut(cmap_file)
            
            # Lightness correction only
            try:
                lut_map2 = equalise_colourmap('RGB', lut_map, 'CIE76', 
                                              np.array([1, 0, 0]), 
                                              len(lut_map) / 25, False, False)
                
                # Save in specified format(s)
                if save_as is None:
                    # Save with same extension as input file
                    if ext == '.mat':
                        try:
                            import scipy.io
                            scipy.io.savemat(output_dir / f'{base_name}.mat', {'map': lut_map2})
                        except ImportError:
                            print(f"  ⚠ scipy required for .mat format, skipping {base_name}")
                            continue
                    elif ext == '.lut':
                        save_lut(output_dir, '', base_name, lut_map2)
                    elif ext == '.clut':
                        save_lut(output_dir, '', f'{base_name}.clut', lut_map2)
                    elif ext == '.cmap':
                        save_cmap(output_dir, '', base_name, lut_map2)
                    elif ext in ['.txt', '.csv']:
                        np.savetxt(output_dir / f'{base_name}.csv', lut_map2, delimiter=',')
                else:
                    # Save in specified format only
                    save_fmt = save_as.lower().lstrip('.')
                    if save_fmt == 'mat':
                        try:
                            import scipy.io
                            scipy.io.savemat(output_dir / f'{base_name}.mat', {'map': lut_map2})
                        except ImportError:
                            raise ImportError("scipy is required for .mat format")
                    elif save_fmt == 'lut':
                        save_lut(output_dir, '', base_name, lut_map2)
                    elif save_fmt == 'clut':
                        save_lut(output_dir, '', f'{base_name}.clut', lut_map2)
                    elif save_fmt == 'cmap':
                        save_cmap(output_dir, '', base_name, lut_map2)
                    elif save_fmt in ['txt', 'csv']:
                        np.savetxt(output_dir / f'{base_name}.csv', lut_map2, delimiter=',')
                    else:
                        raise ValueError(f"Unsupported format: {save_as}. Use 'mat', 'lut', 'clut', 'cmap', 'txt', or 'csv'")
                
                print(f"  ✓ {base_name}")
            except Exception as e:
                print(f"  ✗ {base_name}: {e}")
                
        except Exception as e:
            print(f"  ✗ Error loading {cmap_file.name}: {e}")
    
    print(f"\nProcessing complete. Output saved to {output_dir}")


def map_isoluminance(path_to_maps: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     save_as: Optional[str] = None) -> None:
    """
    Create isoluminant versions of colour maps.
    
    This function reads colour maps from a directory, applies full isoluminant
    correction (W=[1, 1, 1]), and saves corrected versions.
    
    Parameters
    ----------
    path_to_maps : str or Path
        Path to directory containing colormap files (.mat, .lut, .clut, .cmap, .txt, .csv)
    output_dir : str or Path, optional
        Output directory. Defaults to 'isoluminant_maps' in current directory
    save_as : str, optional
        Format to save: 'mat', 'lut', 'clut', 'cmap', 'txt', 'csv'
        If None, saves with same extension as input files (default: None)
    """
    path_to_maps = Path(path_to_maps)
    
    if output_dir is None:
        output_dir = Path.cwd() / 'isoluminant_maps'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all supported colormap files
    colormap_files = []
    for ext in ['*.lut', '*.clut', '*.cmap', '*.mat', '*.txt', '*.csv']:
        colormap_files.extend(list(path_to_maps.glob(ext)))
    
    if not colormap_files:
        print(f"No colormap files (.lut, .clut, .cmap, .mat, .txt, .csv) found in {path_to_maps}")
        return
    
    print(f"Processing {len(colormap_files)} colour maps for isoluminance...")
    
    for cmap_file in colormap_files:
        try:
            # Load the colour map based on file type
            base_name = cmap_file.stem
            ext = cmap_file.suffix.lower()
            
            if ext == '.mat':
                try:
                    import scipy.io
                    mat_data = scipy.io.loadmat(cmap_file)
                    # Try common variable names
                    if 'colormap' in mat_data:
                        lut_map = mat_data['colormap']
                    elif 'cmap' in mat_data:
                        lut_map = mat_data['cmap']
                    else:
                        # Get first non-metadata variable
                        keys = [k for k in mat_data.keys() if not k.startswith('__')]
                        if keys:
                            lut_map = mat_data[keys[0]]
                        else:
                            raise ValueError("No colormap data found in .mat file")
                except ImportError:
                    print(f"  ⚠ {cmap_file.name}: scipy required for .mat files")
                    continue
            elif ext in ['.txt', '.csv']:
                lut_map = np.loadtxt(cmap_file, delimiter=',')
            else:
                # .lut or .cmap
                lut_map = load_lut(cmap_file)
            
            # Isoluminant correction (full formula)
            try:
                lut_map2_iso = equalise_colourmap('RGB', lut_map, 'CIE76',
                                                  np.array([1, 1, 1]),
                                                  len(lut_map) / 25, False, False)
                
                # Save in specified format(s)
                if save_as is None:
                    # Save with same extension as input file
                    if ext == '.mat':
                        try:
                            import scipy.io
                            scipy.io.savemat(output_dir / f'{base_name}_iso.mat', {'map': lut_map2_iso})
                        except ImportError:
                            print(f"  ⚠ scipy required for .mat format, skipping {base_name}")
                            continue
                    elif ext == '.lut':
                        save_lut(output_dir, '', f'{base_name}_iso', lut_map2_iso)
                    elif ext == '.clut':
                        save_lut(output_dir, '', f'{base_name}_iso.clut', lut_map2_iso)
                    elif ext == '.cmap':
                        save_cmap(output_dir, '', f'{base_name}_iso', lut_map2_iso)
                    elif ext in ['.txt', '.csv']:
                        np.savetxt(output_dir / f'{base_name}_iso.csv', lut_map2_iso, delimiter=',')
                else:
                    # Save in specified format only
                    save_fmt = save_as.lower().lstrip('.')
                    if save_fmt == 'mat':
                        try:
                            import scipy.io
                            scipy.io.savemat(output_dir / f'{base_name}_iso.mat', {'map': lut_map2_iso})
                        except ImportError:
                            raise ImportError("scipy is required for .mat format")
                    elif save_fmt == 'lut':
                        save_lut(output_dir, '', f'{base_name}_iso', lut_map2_iso)
                    elif save_fmt == 'clut':
                        save_lut(output_dir, '', f'{base_name}_iso.clut', lut_map2_iso)
                    elif save_fmt == 'cmap':
                        save_cmap(output_dir, '', f'{base_name}_iso', lut_map2_iso)
                    elif save_fmt in ['txt', 'csv']:
                        np.savetxt(output_dir / f'{base_name}_iso.csv', lut_map2_iso, delimiter=',')
                    else:
                        raise ValueError(f"Unsupported format: {save_as}. Use 'mat', 'lut', 'clut', 'cmap', 'txt', or 'csv'")
                
                print(f"  ✓ {base_name}_iso")
            except Exception as e:
                print(f"  ✗ {base_name}_iso: {e}")
                
        except Exception as e:
            print(f"  ✗ Error loading {cmap_file.name}: {e}")
    
    print(f"\nIsoluminant processing complete. Output saved to {output_dir}")


def equalise_colourmap(rgblab: str, colormap: np.ndarray, 
                       formula: str = 'CIE76',
                       W: Optional[np.ndarray] = None,
                       sigma: float = 0,
                       cyclic: bool = False,
                       diagnostics: bool = False) -> np.ndarray:
    """
    Equalise colour contrast over a colourmap.
    
    Parameters
    ----------
    rgblab : str
        'RGB' or 'LAB' indicating the type of data in colormap
    colormap : np.ndarray
        A Nx3 RGB or CIE L*a*b* colour map
    formula : str
        'CIE76' or 'CIEDE2000'
    W : np.ndarray, optional
        Weight vector for lightness, chroma/a, hue/b components. 
        Defaults to [1, 0, 0] (lightness only)
    sigma : float
        Gaussian smoothing parameter. Suggested: 5-7
    diagnostics : bool
        If True, print diagnostic information
        
    Returns
    -------
    np.ndarray
        RGB colour map with equalised perceptual contrast
        
    References
    ----------
    Peter Kovesi. Good Colour Maps: How to Design Them.
    arXiv:1509.03700 [cs.GR] 2015
    """
    if W is None:
        W = np.array([1, 0, 0])
    else:
        W = np.asarray(W)
        
    N = colormap.shape[0]
    
    if sigma > 0 and N / sigma < 25:
        warnings.warn("It is not recommended that sigma be larger than 1/25 of colour map length")
    
    # Convert to Lab if needed
    if rgblab.upper() == 'RGB':
        if colormap.max() > 1.01 or colormap.min() < -0.01:
            raise ValueError("If map is RGB, values should be in range 0-1")
        lab_map = rgb_to_lab(colormap)
        rgb_map = colormap
    elif rgblab.upper() == 'LAB':
        if np.abs(colormap).max() < 10:
            raise ValueError("If map is LAB, magnitude of values expected to be > 10")
        lab_map = colormap
        rgb_map = lab_to_rgb(colormap)
    else:
        raise ValueError("Input must be RGB or LAB")
    
    L = lab_map[:, 0].copy()
    a = lab_map[:, 1].copy()
    b = lab_map[:, 2].copy()
    
    # Iteratively equalise the colour map
    for iteration in range(3):
        # Compute perceptual colour difference
        if formula.upper() == 'CIE76':
            deltaE = cie76(L, a, b, W)
        elif formula.upper() == 'CIEDE2000':
            deltaE = ciede2000_map(L, a, b, W)
        else:
            raise ValueError("Unknown colour difference formula")
        
        # Ensure all values > 0.001
        deltaE[deltaE < 0.001] = 0.001
        cum_dE = np.cumsum(deltaE)
        
        # Form equal steps in cumulative contrast
        equi_cum_dE = np.linspace(cum_dE[0], cum_dE[-1], N)
        
        # Interpolate to find locations for equal deltaE
        indices = np.arange(N)
        new_indices = np.interp(equi_cum_dE, cum_dE, indices)
        
        # Interpolate Lab values
        L = np.interp(new_indices, indices, L)
        a = np.interp(new_indices, indices, a)
        b = np.interp(new_indices, indices, b)
    
    # Apply smoothing if requested
    if sigma > 0:
        L = smooth(L, sigma)
        a = smooth(a, sigma)
        b = smooth(b, sigma)
    
    # Convert back to RGB
    new_lab_map = np.column_stack([L, a, b])
    new_rgb_map = lab_to_rgb(new_lab_map)
    
    if diagnostics:
        print(f"Input map range: [{rgb_map.min():.3f}, {rgb_map.max():.3f}]")
        print(f"Output map range: [{new_rgb_map.min():.3f}, {new_rgb_map.max():.3f}]")
        print(f"Lab range: L=[{L.min():.1f}, {L.max():.1f}], "
              f"a=[{a.min():.1f}, {a.max():.1f}], "
              f"b=[{b.min():.1f}, {b.max():.1f}]")
    
    return new_rgb_map

def deltaE2000(lab_std: np.ndarray, lab_sample: np.ndarray, 
               klch: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the CIEDE2000 color-difference between reference and sample colors.
    
    Parameters
    ----------
    lab_std : np.ndarray
        Reference color(s) in CIE L*a*b* format. Shape: (N, 3) or (3,)
    lab_sample : np.ndarray
        Sample color(s) in CIE L*a*b* format. Shape: (N, 3) or (3,)
    klch : np.ndarray, optional
        Parametric weighting factors [kL, kC, kH]. Defaults to [1, 1, 1]
        
    Returns
    -------
    np.ndarray
        Color difference values using CIEDE2000 formula
        
    References
    ----------
    Based on "The CIEDE2000 Color-Difference Formula: Implementation Notes,
    Supplementary Test Data, and Mathematical Observations," G. Sharma,
    W. Wu, E. N. Dalal, Color Research and Application, vol. 30. No. 1,
    pp. 21-30, February 2005.
    """
    # Ensure inputs are 2D arrays
    if lab_std.ndim == 1:
        lab_std = lab_std.reshape(1, -1)
    if lab_sample.ndim == 1:
        lab_sample = lab_sample.reshape(1, -1)
        
    # Validate dimensions
    if lab_std.shape != lab_sample.shape:
        raise ValueError("Standard and Sample sizes do not match")
    if lab_std.shape[1] != 3:
        raise ValueError("Lab vectors should be Nx3 arrays")
        
    # Set parametric factors
    if klch is None:
        kl, kc, kh = 1.0, 1.0, 1.0
    else:
        if klch.shape != (3,):
            raise ValueError("KLCH must be a 1x3 vector")
        kl, kc, kh = klch[0], klch[1], klch[2]
    
    # Extract L, a, b components
    L_std = lab_std[:, 0]
    a_std = lab_std[:, 1]
    b_std = lab_std[:, 2]
    Cab_std = np.sqrt(a_std**2 + b_std**2)
    
    L_sample = lab_sample[:, 0]
    a_sample = lab_sample[:, 1]
    b_sample = lab_sample[:, 2]
    Cab_sample = np.sqrt(a_sample**2 + b_sample**2)
    
    Cab_mean = (Cab_std + Cab_sample) / 2
    
    G = 0.5 * (1 - np.sqrt(Cab_mean**7 / (Cab_mean**7 + 25**7)))
    
    ap_std = (1 + G) * a_std
    ap_sample = (1 + G) * a_sample
    Cp_sample = np.sqrt(ap_sample**2 + b_sample**2)
    Cp_std = np.sqrt(ap_std**2 + b_std**2)
    
    # Compute product of chromas
    Cp_prod = Cp_sample * Cp_std
    zc_idx = Cp_prod == 0
    
    # Compute hue angles
    hp_std = np.arctan2(b_std, ap_std)
    hp_std = hp_std + 2 * np.pi * (hp_std < 0)
    hp_std[np.abs(ap_std) + np.abs(b_std) == 0] = 0
    
    hp_sample = np.arctan2(b_sample, ap_sample)
    hp_sample = hp_sample + 2 * np.pi * (hp_sample < 0)
    hp_sample[np.abs(ap_sample) + np.abs(b_sample) == 0] = 0
    
    dL = L_sample - L_std
    dC = Cp_sample - Cp_std
    
    # Compute hue difference
    dhp = hp_sample - hp_std
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < -np.pi)
    dhp[zc_idx] = 0
    
    dH = 2 * np.sqrt(Cp_prod) * np.sin(dhp / 2)
    
    # Weighting functions
    Lp = (L_sample + L_std) / 2
    Cp = (Cp_std + Cp_sample) / 2
    
    # Average Hue Computation
    hp = (hp_std + hp_sample) / 2
    hp = hp - (np.abs(hp_std - hp_sample) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zc_idx] = hp_sample[zc_idx] + hp_std[zc_idx]
    
    Lpm50_2 = (Lp - 50)**2
    Sl = 1 + 0.015 * Lpm50_2 / np.sqrt(20 + Lpm50_2)
    Sc = 1 + 0.045 * Cp
    T = (1 - 0.17 * np.cos(hp - np.pi/6) + 
         0.24 * np.cos(2*hp) + 
         0.32 * np.cos(3*hp + np.pi/30) - 
         0.20 * np.cos(4*hp - 63*np.pi/180))
    Sh = 1 + 0.015 * Cp * T
    
    delta_theta_rad = (30 * np.pi / 180) * np.exp(-((180/np.pi * hp - 275) / 25)**2)
    Rc = 2 * np.sqrt(Cp**7 / (Cp**7 + 25**7))
    RT = -np.sin(2 * delta_theta_rad) * Rc
    
    kl_Sl = kl * Sl
    kc_Sc = kc * Sc
    kh_Sh = kh * Sh
    
    # CIEDE2000 color difference
    de00 = np.sqrt((dL / kl_Sl)**2 + 
                   (dC / kc_Sc)**2 + 
                   (dH / kh_Sh)**2 + 
                   RT * (dC / kc_Sc) * (dH / kh_Sh))
    
    return de00


def cie76(L: np.ndarray, a: np.ndarray, b: np.ndarray, 
          W: np.ndarray) -> np.ndarray:
    """
    Compute CIE76 color difference along a color map.
    
    Parameters
    ----------
    L, a, b : np.ndarray
        Lightness and color components
    W : np.ndarray
        Weight vector [wL, wa, wb] for components
        
    Returns
    -------
    np.ndarray
        Delta E values between consecutive entries
    """
    N = len(L)
    deltaE = np.zeros(N)
    
    for i in range(1, N):
        dL = L[i] - L[i-1]
        da = a[i] - a[i-1]
        db = b[i] - b[i-1]
        deltaE[i] = np.sqrt((W[0]*dL)**2 + (W[1]*da)**2 + (W[2]*db)**2)
    
    return deltaE


def ciede2000_map(L: np.ndarray, a: np.ndarray, b: np.ndarray, 
                  W: np.ndarray) -> np.ndarray:
    """
    Compute CIEDE2000 color difference along a color map.
    
    Parameters
    ----------
    L, a, b : np.ndarray
        Lightness and color components
    W : np.ndarray
        Weight vector [kL, kC, kH]
        
    Returns
    -------
    np.ndarray
        Delta E values between consecutive entries
    """
    N = len(L)
    deltaE = np.zeros(N)
    
    for i in range(1, N):
        lab_std = np.array([[L[i-1], a[i-1], b[i-1]]])
        lab_sample = np.array([[L[i], a[i], b[i]]])
        deltaE[i] = deltaE2000(lab_std, lab_sample, W)[0]
    
    return deltaE


def smooth(data: np.ndarray, sigma: float, cyclic: bool = False) -> np.ndarray:
    """
    Apply Gaussian smoothing to data.
    
    Parameters
    ----------
    data : np.ndarray
        Data to smooth
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns
    -------
    np.ndarray
        Smoothed data
    """
    if sigma <= 0:
        return data
    return gaussian_filter1d(data, sigma, mode='nearest')


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE L*a*b* color space.
    
    Parameters
    ----------
    rgb : np.ndarray
        RGB values in range [0, 1]. Shape: (N, 3)
        
    Returns
    -------
    np.ndarray
        L*a*b* values. L in [0, 100], a and b approximately in [-128, 127]
    """
    # Convert RGB to XYZ (using D65 illuminant)
    # First apply gamma correction
    rgb = np.asarray(rgb)
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055)**2.4, rgb / 12.92)
    
    # RGB to XYZ transformation matrix (sRGB with D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    xyz = rgb_linear @ M.T
    
    # XYZ to Lab (D65 white point)
    xyz_n = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / xyz_n
    
    mask = xyz > 0.008856
    f_xyz = np.where(mask, xyz**(1/3), (7.787 * xyz) + (16/116))
    
    L = 116 * f_xyz[:, 1] - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
    
    return np.column_stack([L, a, b])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE L*a*b* to RGB color space.
    
    Parameters
    ----------
    lab : np.ndarray
        L*a*b* values. Shape: (N, 3)
        
    Returns
    -------
    np.ndarray
        RGB values in range [0, 1]
    """
    lab = np.asarray(lab)
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    
    # Lab to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Inverse transformation
    xyz = np.column_stack([fx, fy, fz])
    mask = xyz > 0.206897  # (6/29)^3
    xyz = np.where(mask, xyz**3, (xyz - 16/116) / 7.787)
    
    # Apply D65 white point
    xyz_n = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz * xyz_n
    
    # XYZ to RGB
    M_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [0.0556434, -0.2040259,  1.0572252]])
    
    rgb_linear = xyz @ M_inv.T
    
    # Apply gamma correction
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * rgb_linear**(1/2.4) - 0.055, 12.92 * rgb_linear)
    
    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def load_lut(filename: Union[str, Path]) -> np.ndarray:
    """
    Load color lookup table in ImageJ/MRIcron .lut, FSLeyes .cmap, or MRIcroGL .clut format.
    
    Parameters
    ----------
    filename : str or Path
        Path to .lut, .cmap, or .clut file
        
    Returns
    -------
    np.ndarray
        RGB color map with values in range [0, 1]. Shape: (256, 3)
    """
    filename = Path(filename)
    
    if not filename.exists():
        raise FileNotFoundError(f"Unable to find {filename}")
    
    file_size = filename.stat().st_size
    ext = filename.suffix.lower()
    
    # MRIcroGL .clut format (text-based with control points)
    if ext == '.clut':
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse .clut file
        num_nodes = 0
        intensities = []
        rgba_values = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('numnodes='):
                num_nodes = int(line.split('=')[1])
            elif line.startswith('nodeintensity'):
                intensities.append(int(line.split('=')[1]))
            elif line.startswith('nodergba'):
                # Format: nodergba0=R|G|B|A
                rgba_str = line.split('=')[1]
                rgba = [int(x) for x in rgba_str.split('|')]
                rgba_values.append(rgba[:3])  # Take only RGB, ignore alpha
        
        if len(intensities) != len(rgba_values) or len(intensities) == 0:
            raise ValueError(f"Invalid .clut file format: {filename}")
        
        # Convert control points to 256-entry colormap via interpolation
        intensities = np.array(intensities) / 255.0  # Normalize to [0, 1]
        rgba_values = np.array(rgba_values) / 255.0  # Normalize to [0, 1]
        
        # Interpolate to 256 entries
        new_indices = np.linspace(0, 1, 256)
        R = np.interp(new_indices, intensities, rgba_values[:, 0])
        G = np.interp(new_indices, intensities, rgba_values[:, 1])
        B = np.interp(new_indices, intensities, rgba_values[:, 2])
        lut = np.column_stack([R, G, B])
        
        return lut
    
    # ImageJ/MRIcron format: 768 bytes
    if file_size == 768:
        with open(filename, 'rb') as f:
            lut = np.fromfile(f, dtype=np.uint8)
        lut = lut.reshape(256, 3)
        lut = lut / 255.0
        return lut
    
    # FSLeyes .cmap format
    ext = filename.suffix.lower()
    if ext == '.lut' and file_size != 768:
        raise ValueError(f"Unable to read {filename}")
    
    # Read text format
    lut = np.loadtxt(filename)
    if lut.ndim == 1:
        lut = lut.reshape(-1, 3)
    
    if lut.max() > 1.0 or lut.min() < 0:
        raise ValueError(f"RGB should be in range 0..1 for {filename}")
    
    # Ensure 256 entries
    if lut.shape[0] == 255:
        lut = np.vstack([[0, 0, 0], lut])
    
    if lut.shape[0] < 256:
        # Interpolate to 256 entries
        old_indices = np.linspace(0, 1, lut.shape[0])
        new_indices = np.linspace(0, 1, 256)
        R = np.interp(new_indices, old_indices, lut[:, 0])
        G = np.interp(new_indices, old_indices, lut[:, 1])
        B = np.interp(new_indices, old_indices, lut[:, 2])
        lut = np.column_stack([R, G, B])
    
    return lut


def save_lut(path: Union[str, Path], prefix: str, name: str, 
             lut: np.ndarray) -> None:
    """
    Save colormap in ImageJ/MRIcron .lut format or MRIcroGL .clut format.
    
    Parameters
    ----------
    path : str or Path
        Directory to save the file
    prefix : str
        Prefix for filename
    name : str
        Base name for the file (extension determines format)
    lut : np.ndarray
        RGB colormap with values in range [0, 1]. Shape: (N, 3)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Check if name has .clut extension
    name_path = Path(name)
    if name_path.suffix.lower() == '.clut':
        # Save as MRIcroGL .clut format
        base_name = name_path.stem
        filename = path / f"{prefix}{base_name}.clut"
        
        # Create control points (use fewer points for .clut format)
        # Sample at regular intervals
        n_nodes = min(8, len(lut))  # Use up to 8 control points
        indices = np.linspace(0, len(lut) - 1, n_nodes).astype(int)
        
        with open(filename, 'w') as f:
            f.write('[FLT]\n')
            f.write('min=0\n')
            f.write('max=0\n')
            f.write('[INT]\n')
            f.write(f'numnodes={n_nodes}\n')
            f.write('[BYT]\n')
            
            # Write node intensities (0-255)
            for i, idx in enumerate(indices):
                intensity = int(idx * 255 / (len(lut) - 1))
                f.write(f'nodeintensity{i}={intensity}\n')
            
            f.write('[RGBA255]\n')
            
            # Write RGBA values (RGB from lut, A as semi-transparent gradient)
            for i, idx in enumerate(indices):
                r = int(lut[idx, 0] * 255)
                g = int(lut[idx, 1] * 255)
                b = int(lut[idx, 2] * 255)
                a = int((i / (n_nodes - 1)) * 128) if n_nodes > 1 else 128
                f.write(f'nodergba{i}={r}|{g}|{b}|{a}\n')
    else:
        # Save as ImageJ/MRIcron .lut format (binary)
        base_name = name_path.stem
        filename = path / f"{prefix}{base_name}.lut"
        
        # Convert to 0-255 range and save as bytes
        lut_bytes = (lut * 255).astype(np.uint8)
        with open(filename, 'wb') as f:
            lut_bytes.tofile(f)


def save_cmap(path: Union[str, Path], prefix: str, name: str, 
              lut: np.ndarray) -> None:
    """
    Save colormap in FSLeyes .cmap text format.
    
    Parameters
    ----------
    path : str or Path
        Directory to save the file
    prefix : str
        Prefix for filename
    name : str
        Base name for the file
    lut : np.ndarray
        RGB colormap with values in range [0, 1]. Shape: (N, 3)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get base name without extension
    name = Path(name).stem
    
    filename = path / f"{prefix}{name}.cmap"
    
    np.savetxt(filename, lut, fmt='%.6f')




# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the brain colours library.
    """
    import sys
    
    print("Brain Colours - Perceptually Uniform Colour Maps")
    print("=" * 60)
    
    # Example 1: Process colour maps from a directory
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        print(f"\nProcessing colour maps from: {input_path}")
        map_luminance(input_path)
    else:
        print("\nUsage examples:")
        print("-" * 60)
        print("\n1. Process all .lut files in a directory:")
        print("   python make_braincolours.py /path/to/lut_maps/")
        print("\n2. Generate perceptually uniform colour maps:")
        print("""
   from make_braincolours import make_newmap
   import matplotlib.pyplot as plt
   
   # Generate a grey scale
   grey_map, name, desc = make_newmap('grey')  # or make_newmap('L1')
   
   # Generate a heat map
   heat_map, name, desc = make_newmap('heat')  # or make_newmap('L3')
   
   # Generate diverging blue-white-red
   bwr_map, name, desc = make_newmap('bwr')    # or make_newmap('D1')
   
   # Generate diverging blue-grey-yellow (recommended!)
   bgy_map, name, desc = make_newmap('bjy')    # or make_newmap('D7')
   print(f"Generated: {name}")
   print(f"Description: {desc}")
   
   # Use with matplotlib
   plt.imshow(data, cmap=plt.matplotlib.colors.ListedColormap(bgy_map))
        """)
        print("\n3. Use in your code:")
        print("""
   from make_braincolours import (
       load_lut, save_lut, save_cmap, 
       equalise_colourmap, map_luminance,
       deltaE2000, rgb_to_lab, lab_to_rgb, make_newmap
   )
   
   # Load an existing colour map
   mymap = load_lut('mymap.lut')
   
   # Equalise the colour map
   mymap_eq = equalise_colourmap('RGB', mymap, 'CIE76', 
                                [1, 0, 0], 5, False, False)
   
   # Save in various formats
   save_lut('output/', '', 'mymap_eq', mymap_eq)
   save_cmap('output/', '', 'mymap_eq', mymap_eq)
   
   # Process a whole directory
   map_luminance('/path/to/maps/', 'output_dir/')
        """)
        print("\n4. Compute colour difference:")
        print("""
   import numpy as np
   from make_braincolours import deltaE2000
   
   # Two colours in CIE L*a*b* format
   color1 = np.array([[50, 20, 30]])
   color2 = np.array([[55, 25, 35]])
   
   # Compute CIEDE2000 difference
   diff = deltaE2000(color1, color2)
   print(f"Colour difference: {diff[0]:.2f}")
        """)
        print("\n5. Convert between RGB and Lab:")
        print("""
   import numpy as np
   from make_braincolours import rgb_to_lab, lab_to_rgb
   
   # RGB to Lab
   rgb = np.array([[1.0, 0.0, 0.0]])  # Pure red
   lab = rgb_to_lab(rgb)
   print(f"RGB {rgb[0]} -> Lab {lab[0]}")
   
   # Lab to RGB
   rgb_back = lab_to_rgb(lab)
   print(f"Lab {lab[0]} -> RGB {rgb_back[0]}")
        """)
        print("\n" + "=" * 60)

