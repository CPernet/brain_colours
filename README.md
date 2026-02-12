# Brain Colours

Perceptually uniform colour maps for brain imaging visualization, based on the paper [Data visualization for inference in tomographic brain imaging](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.14430) EJN 2019.

## Brain colours maps for usage

24 maps linear or diverging with a linear in luminance + the brewcolor map for categorical data (colorblind friendly).  
All 25 maps come as `.mat .lut .clut .cmap .json`  

## Other folders

I linearized maps of the faviourite tools, you can simply overwrite the maps by those ones. 
Note you may not want all maps linearized, this can be checked in the png files looking at pre- post- (under code/software_linearization)

## Quick Start (Python)

```python
from make_braincolours import make_newmap, map_luminance, map_isoluminance

# Generate perceptually uniform colormaps
cmap, name, desc = make_newmap('grey')    # Linear grey scale
cmap, name, desc = make_newmap('BWR')     # Diverging blue-white-red
cmap, name, desc = make_newmap('BGY')     # Diverging blue-grey-yellow

# Save to file (optional second argument)
cmap, name, desc = make_newmap('blue', save_as='my_colormap.lut')
cmap, name, desc = make_newmap('BGR', save_as='diverging.mat')

# Non-continuous colormaps for discrete/categorical data
cmap_5, name, desc = make_newmap('5_colours')    # 5 distinct colours
cmap_12, name, desc = make_newmap('12_colours')  # 12 distinct colours

# Process existing colormaps - fix luminance issues
map_luminance(path_to_maps='colormaps_dir')                           # Default output
map_luminance(path_to_maps='colormaps_dir', output_dir='corrected')   # Custom output
map_luminance(path_to_maps='colormaps_dir', output_dir='corrected', save_as='lut')  # Only LUT format

# Create isoluminant versions of colormaps
map_isoluminance(path_to_maps='colormaps_dir')                        # Default output
map_isoluminance(path_to_maps='colormaps_dir', output_dir='iso_maps')
map_isoluminance(path_to_maps='colormaps_dir', output_dir='iso_maps', save_as='csv')
```

## Main Features

### 1. `make_newmap(label, save_as=None)`

Generate perceptually uniform colour maps for brain imaging.

**Parameters:**
- `label` (str): Colormap label - see available options below
- `save_as` (str, optional): File path to save. Extension determines format: `.mat`, `.lut`, `.cmap`, `.txt`, `.csv`

**Returns:**
- `colormap` (ndarray): RGB colormap with values in [0, 1]
- `name` (str): Descriptive name
- `desc` (str): Brief description

**Available Colormaps:**

*Linear colormaps (256 colours):*
- `'grey'` or `'L1'` - Grey scale
- `'BRYW'` or `'L3'` - Black-Red-Yellow-White heat map
- `'BRY'` or `'L4'` - Black-Red-Yellow heat map
- `'blue'` or `'L6'` - Blue shades

*Diverging colormaps (256 colours):*
- `'BWR'` or `'D1'` - Blue-White-Red
- `'GWR'` or `'D3'` - Green-White-Red
- `'BGY'` or `'D7'` - Blue-Grey-Yellow
- `'BGR'` or `'D8'` - Blue-Grey-Red

*Non-continuous colormaps (for discrete/categorical data):*
- `'[N]_colours'` where N is an integer (e.g., `'5_colours'`, `'12_colours'`)
- For N ≤ 11: Uses ColorBrewer qualitative palette (colour-blind safe)
- For N > 11: Samples from xrain colour scale at equidistant points

### 2. `map_luminance(path_to_maps, output_dir=None, save_as=None)`

Fix luminance issues in existing colour maps using lightness-only correction.

**Parameters:**
- `path_to_maps` (str or Path): Directory containing colormap files (`.mat`, `.lut`, `.clut`, `.cmap`, `.txt`, `.csv`)
- `output_dir` (str or Path, optional): Output directory. Default: `'new_braincolour_maps'`
- `save_as` (str, optional): Format to save: `'mat'`, `'lut'`, `'clut'`, `'cmap'`, `'txt'`, `'csv'`. If None, saves with same extension as input files

**Notes:**
- Uses W=[1, 0, 0] for lightness-only correction
- Preserves original color characteristics while fixing perceptual uniformity

### 3. `map_isoluminance(path_to_maps, output_dir=None, save_as=None)`

Create isoluminant versions of colour maps with full perceptual correction.

**Parameters:**
- `path_to_maps` (str or Path): Directory containing colormap files (`.mat`, `.lut`, `.clut`, `.cmap`, `.txt`, `.csv`)
- `output_dir` (str or Path, optional): Output directory. Default: `'isoluminant_maps'`
- `save_as` (str, optional): Format to save: `'mat'`, `'lut'`, `'clut'`, `'cmap'`, `'txt'`, `'csv'`. If None, saves with same extension as input files

**Notes:**
- Uses W=[1, 1, 1] for full isoluminant correction
- Creates versions with uniform perceptual steps across all color dimensions
- Outputs include `_iso` suffix

## File Formats

Supported formats for brain imaging software:
- `.mat` - MATLAB format (requires scipy)
- `.lut` - ImageJ/MRIcron binary format (768 bytes)
- `.clut` - MRIcroGL text format with control points
- `.cmap` - FSLeyes text format
- `.txt` / `.csv` - Plain text CSV format

## Background

Colour maps used in scientific literature must reflect the underlying data. The problem with many maps is that while underlying values are represented by different colours (as red-green-blue values) they are not perceived as such. This is explained by the fact that humans do not perceive colours in a uniform fashion, and that most colour maps have colour and luminance confounded. One solution is to make maps in CIELAB (Luminance A:red-green B:blue-yellow) space. This repository provides usual maps but with luminance fixed (linearized as much as possible) based on Peter Kovesi's paper and code: 'Good Colour Maps: How to Design Them' [arXiv:1509.03700](https://arxiv.org/abs/1509.03700)

![alt text](https://github.com/CPernet/brain_colours/blob/master/examples.jpg)

## Data and Code

This repository contains both Python and MATLAB code to generate colour maps, as well as pre-generated maps in .csv, .mat (MATLAB/SPM), .cmap (FSLeyes) and .lut (MRIcron, ImageJ) formats. Brain colours data are made available under the [Open Data Commons Attribution License](http://opendatacommons.org/licenses/by/1.0).

## Installation

### Python
```bash
# Clone the repository
git clone https://github.com/CPernet/brain_colours.git
cd brain_colours/code

# Install dependencies
pip install numpy scipy
```

**Requirements:**
- Python 3.7+
- numpy >= 1.19.0
- scipy >= 1.5.0 (optional, only for .mat file saving)

### MATLAB
The original MATLAB code is in the `code/` directory and requires the dependencies listed below.

## Dependencies

**Python implementation:**
- Color space conversions based on CIE L*a*b* with D65 illuminant
- Perceptual uniformity using CIEDE2000 color difference formula
- B-spline interpolation via scipy

**MATLAB implementation:**
- Initial colour maps are taken from [MRIcron](https://www.nitrc.org/projects/mricron)
- The luminance is fixed using [equalisecolourmap.m](https://www.peterkovesi.com/matlabfns/index.html#colour) and related files provided by Peter Kovesi. Note this also depends on [deltaE2000.m](http://www2.ece.rochester.edu/~gsharma/ciede2000/).
- The visualization is done using RGB to LAB space conversion using the colorspace.m and colormapline.m functions from [Matteo Niccoli](https://mycarta.wordpress.com/2012/05/12/the-rainbow-is-dead-long-live-the-rainbow-part-1/)
