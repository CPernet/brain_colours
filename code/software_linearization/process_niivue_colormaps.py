"""
Process niivue colormaps with luminance correction.

This script:
1. Clones the niivue repository (if not already cloned)
2. Reads all .json colormap files from niivue
3. Applies map_luminance correction
4. Saves corrected versions to a new directory
5. Plots before/after comparison

Cyril Pernet 2026
"""

import numpy as np
from pathlib import Path
import subprocess
import sys
import shutil

# Import from make_braincolours
from make_braincolours import map_luminance

# Import plotting function
from plot_colormaps import plot_colormap_comparison


def clone_niivue_repo(target_dir: str = '/tmp/niivue') -> Path:
    """
    Clone or update niivue repository.
    
    Parameters
    ----------
    target_dir : str
        Directory to clone the repository to
        
    Returns
    -------
    Path
        Path to the niivue repository
    """
    target_path = Path(target_dir)
    
    if target_path.exists():
        print(f"\nNiivue repository already exists at: {target_path}")
        print("  Using existing clone...")
    else:
        print(f"\nCloning niivue repository to: {target_path}")
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/niivue/niivue.git', str(target_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print("  ✓ Clone successful")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to clone repository: {e}")
            sys.exit(1)
    
    return target_path


def process_niivue_colormaps(input_dir: str, output_dir: str):
    """
    Apply map_luminance correction to all niivue colormaps.
    
    Parameters
    ----------
    input_dir : str
        Directory containing niivue .json colormap files
    output_dir : str
        Directory to save corrected colormaps
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing niivue colormaps...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Apply map_luminance (this will process all .json files in the directory)
    map_luminance(input_dir, output_dir, save_as=None)
    
    print(f"\n✓ Luminance correction complete")


def plot_comparison(before_dir: str, after_dir: str, 
                   output_file: str = 'niivue_colormaps_comparison.png',
                   max_plots: int = None):
    """
    Plot before/after comparison of niivue colormaps.
    
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
    
    # Get all .json files from the output directory
    after_files_paths = sorted(after_path.glob('*.json'))
    
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
        print("Warning: No matching .json files found for comparison")
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
    print("Processing Niivue Colormaps")
    print("="*60)
    
    # Step 1: Clone niivue repository
    print("\n[1/3] Setting up niivue repository...")
    niivue_path = clone_niivue_repo()
    
    # Define directories
    input_dir = niivue_path / 'packages' / 'niivue' / 'src' / 'cmaps'
    output_dir = Path('/indirect/staff/cyrilpernet/brain_colours/niivue')
    
    if not input_dir.exists():
        print(f"\n✗ Error: Colormap directory not found: {input_dir}")
        print("  The niivue repository structure may have changed.")
        sys.exit(1)
    
    # Count input files
    input_files = list(input_dir.glob('*.json'))
    print(f"      Found {len(input_files)} colormap files")
    
    # Step 2: Apply luminance correction
    print("\n[2/3] Applying luminance correction...")
    process_niivue_colormaps(str(input_dir), str(output_dir))
    
    # Count corrected files
    corrected_files = list(output_dir.glob('*.json'))
    print(f"      Created {len(corrected_files)} corrected files")
    
    # Step 3: Create comparison plots
    print("\n[3/3] Creating comparison plots...")
    plot_comparison(str(input_dir), str(output_dir))
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nNiivue repository: {niivue_path}")
    print(f"Original colormaps: {input_dir}")
    print(f"Corrected colormaps: {output_dir}")
    print(f"Comparison plot: niivue_colormaps_comparison.png")
    print()


if __name__ == '__main__':
    main()
