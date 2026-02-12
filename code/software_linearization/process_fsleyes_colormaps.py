"""
Process FSLeyes colormaps with luminance correction.

This script:
1. Reads all .cmap files from FSLeyes assets directory
2. Applies map_luminance correction
3. Saves corrected versions to a new directory
4. Plots before/after comparison

Cyril Pernet 2026
"""

import numpy as np
from pathlib import Path
import sys

# Import from make_braincolours
from make_braincolours import map_luminance

# Import plotting function
from plot_colormaps import plot_colormap_comparison


def process_fsleyes_colormaps(input_dir: str, output_dir: str):
    """
    Apply map_luminance correction to all FSLeyes colormaps.
    
    Parameters
    ----------
    input_dir : str
        Directory containing FSLeyes .cmap files
    output_dir : str
        Directory to save corrected colormaps
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing FSLeyes colormaps...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Apply map_luminance (this will process all .cmap files in the directory)
    map_luminance(input_dir, output_dir, save_as=None)
    
    print(f"\n✓ Luminance correction complete")


def plot_comparison(before_dir: str, after_dir: str, output_file: str = 'fsleyes_colormaps_comparison.png'):
    """
    Plot before/after comparison of selected colormaps.
    
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
    
    # Get all .cmap files from the output directory
    after_files_paths = sorted(after_path.glob('*.cmap'))
    
    before_files = []
    after_files = []
    
    # Find matching before files for each after file
    for after_file in after_files_paths:
        before_file = before_path / after_file.name
        
        if before_file.exists():
            before_files.append(str(before_file))
            after_files.append(str(after_file))
    
    if not before_files or not after_files:
        print("Warning: No matching .cmap files found for comparison")
        return
    
    print(f"\nGenerating comparison plot for {len(before_files)} colormaps...")
    
    # Calculate appropriate figure size based on number of colormaps
    # Each colormap row needs about 1.5 inches height
    fig_height = max(12, len(before_files) * 1.5)
    
    # Create comparison plot
    plot_colormap_comparison(
        before_files,
        after_files,
        output_file=output_file,
        figsize=(14, fig_height)
    )
    
    print(f"✓ Comparison plot saved to: {output_file}")


def main():
    """Main execution function."""
    print("="*60)
    print("Processing FSLeyes Colormaps")
    print("="*60)
    
    # Define directories
    input_dir = '/usr/local/fsl6.0.7/lib/python3.11/site-packages/fsleyes/assets/colourmaps'
    output_dir = '/indirect/staff/cyrilpernet/brain_colours/fsleyes'
    
    # Step 1: Apply luminance correction
    print("\n[1/2] Applying luminance correction...")
    process_fsleyes_colormaps(input_dir, output_dir)
    
    # Count corrected files
    corrected_files = list(Path(output_dir).glob('*.cmap'))
    print(f"      Created {len(corrected_files)} corrected files")
    
    # Step 2: Create comparison plots
    print("\n[2/2] Creating comparison plots...")
    plot_comparison(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nOriginal colormaps: {input_dir}")
    print(f"Corrected colormaps: {output_dir}")
    print(f"Comparison plot: fsleyes_colormaps_comparison.png")
    print()


if __name__ == '__main__':
    main()
