# SaGEA Toolbox
SaGEA (Satellite Gravity error assessment) is a Python-based project for comprehensive error assessment of GRACE and GRACE-FO based mass change.
This toolbox also comes with post-processing functions of GRACE(-FO)'s level-2 products,
as well as the collection of the level-2 products, and the functions of the results visualization.

## Installation
This program homepage is: https://github.com/NCSGgroup/SaGEA.

Use this code to download this project.

`git clone https://github.com/NCSGgroup/SaGEA`

This project is developed based on Python 3.9 and the dependencies are listed in `requirements.txt`.

Use this code to download the dependencies.

`pip install -r requirements.txt`

## Features
- Auto-collecting GRACE(-FO) level-2 products and related auxiliary files.
- Complete and diverse popular methodologies and technologies of GRACE(-FO)'s post-processing.
- Types of Error assessment/quantification of GRACE(-FO) based mass change.
- User interface (under construction).

## Module Description and Usage

### Data Class: SHC

### Data Class: GRID

### Data Collecting
GRACE and GRACE-FO level-2 products can be obtained at open source FTP server ftp://isdcftp.gfz-potsdam.de/.
Level-2 products includes GSM, GAA, GAB, GAC, and GAD (The last four products are collectively referred to as GAX.),
which are given in fully normalized spherical harmonic coefficients (SHCs) of gravity potential.

- GSM products represent the estimate of Earth's mean gravity field during the specified timespan derived from GRACE mission measurements.

- GAA products represent the anomalous contributions of the non-tidal atmosphere to the Earth's mean gravity field during the specified timespan.

- GAB products represent the anomalous contributions of the non-tidal dynamic ocean to ocean bottom pressure during the specified timespan.

- GAC products represent the sum of the GAA and GAB coefficients during the specified timespan.

- GAD products give the SHCs that are zero over the continents, and provide the anomalous simulated ocean bottom pressure that includes non-tidal air and water contributions elsewhere during the specified timespan.

The most used GSM solutions are given by three processing centers, that is, Center for Space Research (CSR), University of Texas at Austin, Jet Propulsion Laboratory (JPL), NASA, and German Research for Geosciences (GFZ), German.

Path `/pysrc/data_collection/` includes the source file to access remote servers and download GRACE level-2 products from the above FTP server,
including the above GSM, GAX, and necessary low-degree files for replacing and other auxiliary files.
It is recommended to control this program through an external configuration file.

A demo program `/demo/data_collecting/demoCollectL2Data.py` gives an example to download GRACE level-2 files by a configuration file `/setting/data_collecting/CollectL2Data.json`.
Users can simply modify the parameters in the configuration file and run the above program to achieve automatic collection of the corresponding files.

This demo program also gives an example to download the low-degree files. Users can simply run the above program to achieve automatic collections.

### Loading local GRACE level-2 products and replacing low-degree coefficients
GRACE level-2 GSM solutions lack the three degree-1 coefficients,
which are proportional to geocenter motion and can not been ignored for a complete representation of the mass redistribution in the Earth system (Sun et al., 2016).
The GRACE-based C20 coefficient is subject to large uncertainties (Chen et al., 2016).
Besides, The C30 coefficient is also shown to be poorly observed by GRACE/GRACE-FO when either mission is operating without two fully functional accelerometers (Loomis et al., 2020).
Before using the GSM solutions,
the degree-1 coefficients needs to be added back, 
and C20, C30 coefficients with large uncertainties also needs to be replaced with estimates from other techniques, 
such as satellite laser ranging (SLR).

Path `/pysrc/auxiliary/load_file/LoadL2SH.py` provides functions to load GRACE level-2 products,
and path `/pysrc/auxiliary/load_file/LoadL2LowDeg.py` provides those of the low-degree coefficient.
Path `/pysrc/auxiliary/load_file/replace_low_degree/` includes the source file to apply the replacing low-degree coefficients on given SHCs Cqlm and Sqlm.

Note that in SGPP Toolbox, all the numerical calculation shall be completed in the form of vectorization if possible, so the SH coefficients are always represented as two 3d arrays, Cqlm and Sqlm, where q stands for the number of data sets, l, m stands for the degree and order, separately. Even if there is only one set data to deal with, it still requires to refer to the SH data as two 3d arrays in shape of (1, l, m).

A demo program SGPPToolbox/demo/demo_laod_GSM_and_replace.py gives a simple example to load GRACE level-2 products and low-degree products. And corresponding replacement can also get applied in this demo as users need. For more information, please refer to the function introduction in the demo file.

Unit conversion and harmonic synthesis/analysis
GRACE level-2 products reflects the distribution of geopotential, from which we can obtain the corresponding changes in mass or other factors through the load theory. As a result, the products are usually get converted to the unit of researcher's interest before further use.

Also, GRACE level-2 products are generally given in the form of SHCs, which can intuitively reflect the signal of different wavelengths (frequency bands). However, it is difficult to directly see in SHCs how the spatial distribution is. Indeed, by performing spherical harmonic synthesis on the SHCs, corresponding grid data can be obtained, from which we can easily see the spatial distribution of signals. On the contrary, the corresponding SHCs can also be obtained through spherical harmonic analysis of grid data.

Path SGPPToolbox/pysrc/convert_field_type/ includes the source files to convert the physical quantity like equivalent water height (EWH), and the required Love number can be obtained by the source files in SGPPToolbox/pysrc/Love_number/.

Path SGPPToolbox/pysrc/harmonic/ includes the source files to do the harmonic synthesis and analysis. and the required associated Legendre polynomial can be obtained by the auxiliary methods in source file SGPPToolbox/pysrc/auxiliary/GeoMathKit.py.

A demo program SGPPToolbox/demo/demo_convert_and_harmonic.py gives a simple example to use the above programs. For more information, please refer to the function introduction in the corresponding source files.

Filtering spherical harmonic coefficients
Due to the existence of high-order noise and correlation error in the GRACE solutions, filtering is a necessary step before apply it on some scientific studies (Wahr et al., 2006). The most usd GRACE filter, isotropic Gaussian filter, was first suggested and applied by Wahr et al. (1998). Other filters based on different principles were raised then like empirical decorrelation filtering (EDF) raised by Wahr et al. (2006), DDK filter by Kusche (2007), etc. Among them, EDF was also improved and used by scholars since its initial proposal (Duan et al., 2009). This toolbox contains the following filtering methods:

Different types of EDFs.
Isotropic Gaussian filter.
Non-isotropic Gaussian fielter by Han et al. (2005).
Fan filter by Zhang et al. (2009).
DDK filter.
Path SGPPToolbox/pysrc/filter/ includes the source files to filter the SHCs. Each filter has different parameter requirements but the same basic usage apply_to(cqlm, [sqlm]) to return the filtered result, and a demo program SGPPToolbox/demo/demo_filter.py gives a simple example to use the above programs. For more information, please refer to the function introduction in the corresponding source files.

Leakage Reduction
Filters for GRACE can supress the noise at high degrees, but at the same time they could also weaken the signal as well. Spatially speaking, signal where it is strong would leak into some place where it is weaker, for example, the hydrological and atmospheric pressure signals over continents will leak into the oceanic estimates at the coastline (Wahr et al., 1998). Depending on the signal strength of the study basin, one would consider to reduce the leak-in signal outside the basin (leakage), or to restore the leak-out signal from the basin (bias).

To reduce the leakage, Wahr et al., 1998 for the first time gave an iterative estimation technique to handle it. Here is a brief introduction to the technique to estimate the leakage:

One gives the initial signal a small-scale filter, and harmonic synthesis it into a pre-smoothed spatial distribution.
The pre-smoothed spatial signal is set zero outside the interested basin, and then get harmonic analyzed, marked as pre-smoothed SHCs outside.
Pre-smoothed SHCs outside then get filtered once again of the same scale with that used in the study, and then get harmonic synthesised it into a new spatial distribution. The new spatial signal inside the interest basin can be seen as the leakage and reduced from the filtered signal.
As for the bias, it is usually to gain a scale factor k to calibrate the filtered or smoothed signal. Researchers have raised numbers of method to estimate k, and other methods to restore correct the bias.

Vishwakarma et al. (2018) listed some commonly used methods to correct the leakage and bias, that is,

Multiplicative (Longuevergne et al., 2007).
Additive (Klees et al., 2007).
Scaling (Landerer et al., 2012).
Data-driven (Vishwakarma et al., 2017).
Besides, forward modeling (FM) by Chen et al. (2015) is another effective and commonly used iterative method to reduce the leakage. One of its characteristics is that it can reconstruct the signal of spatial distribution.

This toolbox contains most of the processing methods mentioned above, Path SGPPToolbox/pysrc/leakage/ includes the source files to handle the leakage error. Like the filters, each method has different parameter requirements but the same basic usage apply_to(gqij) to return the corrected result, and a demo program SGPPToolbox/demo/demo_leakage.py gives a simple example to use the above programs. For more information, please refer to the function introduction in the corresponding source files.