"""
Process standard colormaps (viridis, magma, plasma, turbo) with luminance correction.

This script:
1. Extracts matplotlib viridis, magma, plasma, and turbo colormaps
2. Saves them to disk in multiple formats
3. Applies map_luminance correction
4. Saves corrected versions to a new subfolder
5. Plots before/after comparison

Cyril Pernet 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import from make_braincolours
from make_braincolours import save_lut, save_cmap, map_luminance

# Import plotting function
from plot_colormaps import plot_colormap_comparison


def extract_and_save_colormaps(output_dir: str = '../standard_maps') -> list:
    """
    Extract matplotlib colormaps and save them to disk.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the colormaps
        
    Returns
    -------
    list
        List of saved colormap file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colormaps to extract
    cmap_names = ['viridis', 'magma', 'plasma', 'turbo']
    
    saved_files = []
    
    print(f"\nExtracting {len(cmap_names)} matplotlib colormaps...")
    
    for cmap_name in cmap_names:
        print(f"  ✓ Extracting {cmap_name}...")
        
        # Get colormap from matplotlib
        cmap = plt.get_cmap(cmap_name)
        
        # Sample 256 colors
        colors = cmap(np.linspace(0, 1, 256))[:, :3]  # Remove alpha channel
        
        # Save in multiple formats
        # Save as .lut (binary format)
        lut_file = output_path / f"{cmap_name}.lut"
        save_lut(str(output_path), '', cmap_name, colors)
        saved_files.append(str(lut_file))
        
        # Save as .cmap (text format)
        cmap_file = output_path / f"{cmap_name}.cmap"
        save_cmap(str(output_path), '', cmap_name, colors)
        
        # Save as .csv
        csv_file = output_path / f"{cmap_name}.csv"
        # Convert to 0-255 range for CSV
        colors_255 = (colors * 255).astype(int)
        np.savetxt(csv_file, colors_255, fmt='%d', delimiter=',', 
                   header='R,G,B', comments='')
    
    print(f"\n✓ All colormaps saved to: {output_path}")
    return saved_files


def apply_luminance_correction(input_dir: str, output_dir: str) -> list:
    """
    Apply map_luminance correction to all colormaps.
    
    Parameters
    ----------
    input_dir : str
        Directory containing original colormaps
    output_dir : str
        Directory to save corrected colormaps
        
    Returns
    -------
    list
        List of corrected colormap file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nApplying luminance correction...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Apply map_luminance (this will process all .lut files in the directory)
    map_luminance(input_dir, output_dir, save_as=None)
    
    print(f"✓ Luminance correction complete")
    
    # Return list of corrected files
    corrected_files = sorted(Path(output_dir).glob('*.lut'))
    return [str(f) for f in corrected_files]


def plot_comparison(before_dir: str, after_dir: str, output_file: str = 'standard_colormaps_comparison.png'):
    """
    Plot before/after comparison of colormaps.
    
    Parameters
    ----------
    before_dir : str
        Directory with original colormaps
    after_dir : str
        Directory with corrected colormaps
    output_file : str
        Output filename for the plot
    """
    before_path = Path(before_dir)
    after_path = Path(after_dir)
    
    # Get all .lut files from both directories
    before_files = sorted(before_path.glob('*.lut'))
    after_files = sorted(after_path.glob('*.lut'))
    
    if not before_files or not after_files:
        print("Warning: No .lut files found for comparison")
        return
    
    print(f"\nGenerating comparison plot...")
    print(f"  Comparing {len(before_files)} colormaps")
    
    # Create comparison plot
    plot_colormap_comparison(
        [str(f) for f in before_files],
        [str(f) for f in after_files],
        output_file=output_file
    )
    
    print(f"✓ Comparison plot saved to: {output_file}")


def main():
    """Main execution function."""
    print("="*60)
    print("Processing Standard Colormaps")
    print("="*60)
    
    # Define directories
    base_dir = Path(__file__).parent.parent
    original_dir = base_dir / 'standard_maps'
    corrected_dir = base_dir / 'standard_maps_corrected'
    
    # Step 1: Extract and save colormaps
    print("\n[1/3] Extracting matplotlib colormaps...")
    saved_files = extract_and_save_colormaps(str(original_dir))
    print(f"      Saved {len(saved_files)} colormap files")
    
    # Step 2: Apply luminance correction
    print("\n[2/3] Applying luminance correction...")
    corrected_files = apply_luminance_correction(str(original_dir), str(corrected_dir))
    print(f"      Created {len(corrected_files)} corrected files")
    
    # Step 3: Create comparison plots
    print("\n[3/3] Creating comparison plots...")
    plot_comparison(str(original_dir), str(corrected_dir))
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nOriginal colormaps: {original_dir}")
    print(f"Corrected colormaps: {corrected_dir}")
    print(f"Comparison plot: standard_colormaps_comparison.png")
    print()


if __name__ == '__main__':
    main()
