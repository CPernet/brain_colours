"""
Process SPM slover colormaps with luminance correction.

This script:
1. Reads all .lut and .mat colormap files from SPM slover directory
2. Applies map_luminance correction
3. Saves corrected versions to spm/slover folder
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


def process_spm_colormaps(input_dir: str, output_dir: str):
    """
    Apply map_luminance correction to all SPM slover colormaps.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SPM .lut and .mat files
    output_dir : str
        Directory to save corrected colormaps
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing SPM slover colormaps...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Apply map_luminance (this will process all .lut and .mat files in the directory)
    map_luminance(input_dir, output_dir, save_as=None)
    
    print(f"\n✓ Luminance correction complete")


def plot_comparison(before_dir: str, after_dir: str, 
                   output_file: str = 'spm_slover_colormaps_comparison.png',
                   max_plots: int = None):
    """
    Plot before/after comparison of SPM slover colormaps.
    
    Parameters
    ----------
    before_dir : str
        Directory with original colormaps
    after_dir : str
        Directory with corrected colormaps
    output_file : str
        Output filename for the plot
    max_plots : int, optional
        Maximum number of colormaps to plot. If None, plots all colormaps.
    """
    before_path = Path(before_dir)
    after_path = Path(after_dir)
    
    # Get all .lut and .mat files from the output directory
    after_files_paths = sorted(list(after_path.glob('*.lut')) + list(after_path.glob('*.mat')))
    
    # Limit to max_plots if specified
    if max_plots is not None and len(after_files_paths) > max_plots:
        print(f"\n  Note: Plotting first {max_plots} of {len(after_files_paths)} colormaps")
        after_files_paths = after_files_paths[:max_plots]
    
    before_files = []
    after_files = []
    
    # Find matching before files for each after file
    for after_file in after_files_paths:
        before_file = before_path / after_file.name
        
        if before_file.exists():
            before_files.append(str(before_file))
            after_files.append(str(after_file))
    
    if not before_files or not after_files:
        print("Warning: No matching colormap files found for comparison")
        return
    
    print(f"\nGenerating comparison plot for {len(before_files)} colormaps...")
    
    # Calculate appropriate figure size
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
    print("Processing SPM slover Colormaps")
    print("="*60)
    
    # Define directories
    input_dir = '/users/cyrilpernet/matlab/spm/@slover/private'
    output_dir = '/indirect/staff/cyrilpernet/brain_colours/spm/slover'
    
    # Check if input directory exists
    if not Path(input_dir).exists():
        print(f"\n✗ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Count input files (.lut and .mat files)
    input_lut = list(Path(input_dir).glob('*.lut'))
    input_mat = list(Path(input_dir).glob('*.mat'))
    total_input = len(input_lut) + len(input_mat)
    print(f"\n      Found {total_input} colormap files ({len(input_lut)} .lut, {len(input_mat)} .mat)")
    
    # Step 1: Apply luminance correction
    print("\n[1/2] Applying luminance correction...")
    process_spm_colormaps(input_dir, output_dir)
    
    # Count corrected files
    corrected_lut = list(Path(output_dir).glob('*.lut'))
    corrected_mat = list(Path(output_dir).glob('*.mat'))
    total_corrected = len(corrected_lut) + len(corrected_mat)
    print(f"      Created {total_corrected} corrected files ({len(corrected_lut)} .lut, {len(corrected_mat)} .mat)")
    
    # Step 2: Create comparison plots
    print("\n[2/2] Creating comparison plots...")
    plot_comparison(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nOriginal colormaps: {input_dir}")
    print(f"Corrected colormaps: {output_dir}")
    print(f"Comparison plot: spm_slover_colormaps_comparison.png")
    print()


if __name__ == '__main__':
    main()
