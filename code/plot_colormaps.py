"""
Plot colormaps before and after luminance correction.

This script takes two lists of colormap files (before and after correction)
and plots them side by side, showing the luminance profile of each.
Cyril Pernet 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union
from make_braincolours import load_lut, rgb_to_lab


def plot_colormap_comparison(before_files: List[Union[str, Path]], 
                             after_files: List[Union[str, Path]],
                             output_file: str = 'colormap_comparison.png',
                             figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot colormaps before and after correction side by side.
    
    Parameters
    ----------
    before_files : list of str or Path
        List of colormap file paths before correction
    after_files : list of str or Path
        List of colormap file paths after correction
    output_file : str, optional
        Output filename for the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Examples
    --------
    >>> before = ['original/hot.lut', 'original/jet.lut']
    >>> after = ['corrected/hot.lut', 'corrected/jet.lut']
    >>> plot_colormap_comparison(before, after)
    """
    n_maps = len(before_files)
    if len(after_files) != n_maps:
        raise ValueError("Number of before and after files must match")
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_maps, 2, figsize=figsize)
    if n_maps == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Colormap Comparison: Before vs After Luminance Correction', 
                 fontsize=14, fontweight='bold')
    
    for idx, (before_path, after_path) in enumerate(zip(before_files, after_files)):
        before_path = Path(before_path)
        after_path = Path(after_path)
        
        # Load colormaps
        try:
            cmap_before = load_lut(before_path)
            cmap_after = load_lut(after_path)
        except Exception as e:
            print(f"Error loading {before_path.name} or {after_path.name}: {e}")
            continue
        
        # Convert to Lab and extract luminance
        lab_before = rgb_to_lab(cmap_before)
        lab_after = rgb_to_lab(cmap_after)
        
        L_before = lab_before[:, 0]
        L_after = lab_after[:, 0]
        
        x = np.arange(len(L_before))
        
        # Compute expected linear fit from min to max luminance
        L_min = min(L_before.min(), L_after.min())
        L_max = max(L_before.max(), L_after.max())
        L_expected = np.linspace(L_min, L_max, len(L_before))
        
        # Compute Mean Squared Error vs expected linear
        mse_before = np.mean((L_before - L_expected) ** 2)
        mse_after = np.mean((L_after - L_expected) ** 2)
        
        x = np.arange(len(L_before))
        
        # Plot BEFORE (left column)
        ax_before = axes[idx, 0]
        plot_colormap_line(ax_before, x, L_before, cmap_before)
        # Plot expected linear fit
        ax_before.plot(x, L_expected, 'k--', linewidth=1.5, alpha=0.7, label='Expected linear')
        ax_before.set_ylabel('Luminance (L*)', fontsize=10)
        ax_before.set_title(f'{before_path.stem} MSE = {mse_before:.2f}', fontsize=10, color='red', fontweight='bold')
        ax_before.grid(True, alpha=0.3)
        ax_before.set_ylim([0, 100])
        
        if idx == n_maps - 1:
            ax_before.set_xlabel('Colour level', fontsize=10)
        else:
            ax_before.set_xticklabels([])
        
        # Plot AFTER (right column)
        ax_after = axes[idx, 1]
        plot_colormap_line(ax_after, x, L_after, cmap_after)
        # Plot expected linear fit
        ax_after.plot(x, L_expected, 'k--', linewidth=1.5, alpha=0.7, label='Expected linear')
        ax_after.set_title(f'{after_path.stem} MSE = {mse_after:.2f}', fontsize=10, color='green', fontweight='bold')
        ax_after.grid(True, alpha=0.3)
        ax_after.set_ylim([0, 100])
        ax_after.set_yticklabels([])
        
        if idx == n_maps - 1:
            ax_after.set_xlabel('Colour level', fontsize=10)
        else:
            ax_after.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_file}")
    plt.close()


def plot_colormap_line(ax, x, y, colormap):
    """
"""Plot a line with colors from the colormap (similar to MATLAB colormapline).
    
    Python implementation of colormapline.m by Matthias Hunstig
    (University of Paderborn, Germany, 2013-2016).
    Original MATLAB version: https://www.mathworks.com/matlabcentral/fileexchange/39972
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates (luminance values)
    colormap : np.ndarray
        RGB colormap (N, 3) with values in [0, 1]
    """
    n_colors = len(colormap)
    n_points = len(x)
    
    # Resample colormap if needed
    if n_colors != n_points:
        indices = np.linspace(0, n_colors - 1, n_points).astype(int)
        colors = colormap[indices]
    else:
        colors = colormap
    
    # Plot line segments with individual colors
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2)


def plot_colormap_grid(colormap_files: List[Union[str, Path]],
                       output_file: str = 'colormap_grid.png',
                       ncols: int = 2,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot a grid of colormaps with their luminance profiles.
    
    Parameters
    ----------
    colormap_files : list of str or Path
        List of colormap file paths
    output_file : str, optional
        Output filename for the plot
    ncols : int, optional
        Number of columns in the grid
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    n_maps = len(colormap_files)
    nrows = int(np.ceil(n_maps / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_maps > 1 else [axes]
    
    fig.suptitle('Colormap Luminance Profiles', fontsize=14, fontweight='bold')
    
    for idx, cmap_path in enumerate(colormap_files):
        cmap_path = Path(cmap_path)
        ax = axes[idx]
        
        try:
            # Load colormap
            cmap = load_lut(cmap_path)
            
            # Convert to Lab and extract luminance
            lab = rgb_to_lab(cmap)
            L = lab[:, 0]
            
            x = np.arange(len(L))
            
            # Plot
            plot_colormap_line(ax, x, L, cmap)
            ax.set_title(cmap_path.stem, fontsize=9)
            ax.set_ylabel('Luminance (L*)', fontsize=8)
            ax.set_xlabel('Colour level', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            
        except Exception as e:
            print(f"Error loading {cmap_path.name}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{cmap_path.name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Grid plot saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plot_colormaps.py <before_dir> <after_dir> [map1 map2 ...]")
        print("\nExample:")
        print("  python plot_colormaps.py original/ corrected/ hot jet NIH")
        print("  python plot_colormaps.py /path/to/before/ /path/to/after/ 4hot bone")
        sys.exit(1)
    
    before_dir = Path(sys.argv[1])
    after_dir = Path(sys.argv[2])
    
    if len(sys.argv) > 3:
        # Specific colormap names provided
        map_names = sys.argv[3:]
        before_files = []
        after_files = []
        
        # Search for files with these names
        for name in map_names:
            # Try different extensions
            found_before = None
            found_after = None
            
            for ext in ['.clut', '.lut', '.cmap', '.csv']:
                before_path = before_dir / f'{name}{ext}'
                if before_path.exists():
                    found_before = before_path
                    break
            
            for ext in ['.clut', '.lut', '.cmap', '.csv']:
                after_path = after_dir / f'{name}{ext}'
                if after_path.exists():
                    found_after = after_path
                    break
            
            if found_before and found_after:
                before_files.append(found_before)
                after_files.append(found_after)
            else:
                print(f"⚠ Could not find both before/after for: {name}")
    else:
        # Use all colormaps in before directory
        before_files = sorted(list(before_dir.glob('*.clut')) + 
                             list(before_dir.glob('*.lut')) +
                             list(before_dir.glob('*.cmap')))
        
        after_files = []
        for bf in before_files:
            af = after_dir / bf.name
            if af.exists():
                after_files.append(af)
            else:
                print(f"⚠ No matching after file for: {bf.name}")
        
        # Keep only matching pairs
        before_files = before_files[:len(after_files)]
    
    if before_files and after_files:
        print(f"Plotting {len(before_files)} colormap comparisons...")
        plot_colormap_comparison(before_files, after_files)
    else:
        print("No matching colormap pairs found!")
