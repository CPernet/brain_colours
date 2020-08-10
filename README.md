# Brain colours

code underlying the colormaps in the paper [Data visualization for inference in tomographic brain imaging](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.14430) EJN 2019

## Background

Colour maps used in scientific litterature must reflect the underlying data. The problem with many maps is that while underlying values are represented by different colours (as red-green-blue values) they are not perceived as such. This is explained by the fact that human do not perceive colours in a uniform fashion, and that most colour maps have colour and luminance confounded. One solution is the make map in CIELAB (Luminance A:red-green B:blue-yellow) space. I share here usual maps but with luminance fixed (linearized as much as possible) based on Peter Kovesi paper and code: 'Good Colour Maps: How to Design Them' [arXiv:1509.03700](https://arxiv.org/abs/1509.03700)

![alt text](https://github.com/CPernet/brain_colours/blob/master/examples.jpg)

## Data and Code

This repository contains the Malab code used to generate colour maps as well as the maps as .csv, .mat (Matlab/SPM), .cmap (FSLeyes) and .lut (MRIcron, ImageJ) files. Brain colours data are made available under the [Open Data Commons Attribution License](http://opendatacommons.org/licenses/by/1.0).

## Dependencies

- Initial colour maps are taken from [MRIcron](https://www.nitrc.org/projects/mricron)
- The luminance is fixed using [equalisecolourmap.m](https://www.peterkovesi.com/matlabfns/index.html#colour) and related files provided by Peter Kovesi. Note this also depends on [deltaE2000.m](http://www2.ece.rochester.edu/~gsharma/ciede2000/).
- The vizualization is done using are RGB to LAB space converion using the colorspace.m and colormapline.m functions from [Matteo Niccoli](https://mycarta.wordpress.com/2012/05/12/the-rainbow-is-dead-long-live-the-rainbow-part-1/)
