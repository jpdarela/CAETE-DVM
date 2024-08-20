# Input data for CAETÊ

The pre_processing.py script is used to prepare files that are employed to feed the CAETÊ model.
The raw climatic and edaphic data in these files are publicly available from other sources.

Climatic data: ISIMIP3a/b

Soil data: HWSD, IGBP, & own sources

## References

### ISIMIP climate input

Weedon, G. P., Balsamo, G., Bellouin, N., Gomes, S., Best, M. J., & Viterbo, P. (2014). The WFDEI meteorological forcing data set: WATCH Forcing Data methodology applied to ERA-Interim reanalysis data. Water Resources Research, 50(9), 7505–7514. [https://doi.org/10.1002/2014WR015638](https://doi.org/10.1002/2014WR015638)

Lange, Stefan (2019): EartH2Observe, WFDEI and ERA-Interim data Merged and Bias-corrected for ISIMIP (EWEMBI). V. 1.1. GFZ Data Services. [https://doi.org/10.5880/pik.2019.004](https://doi.org/10.5880/pik.2019.004)

Lange, S., & Büchner, M. (2020). ISIMIP2a atmospheric climate input data. ISIMIP Repository. [https://doi.org/10.48364/ISIMIP.886955](https://doi.org/10.48364/ISIMIP.886955)


The raw input climatic data was downloaded from the [ISIMIP REPOSITORY](https://www.isimip.org/outputdata/isimip-repository/).

### Soil data:

HWSD:
Wieder, W.R., J. Boehnert, G.B. Bonan, and M. Langseth. 2014. Regridded Harmonized World Soil Database v1.2. Data set. Available on-line \[[http://daac.ornl.gov](http://daac.ornl.gov)\] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, USA. [http://dx.doi.org/10.3334/ORNLDAAC/1247](http://dx.doi.org/10.3334/ORNLDAAC/1247)

IGBP: Global Soil Data Task. (2000). Global Gridded Surfaces of Selected Soil Characteristics (IGBP-DIS). Data set. Available on-line \[[http://daac.ornl.gov](http://daac.ornl.gov)\] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, USA. [https://doi.org/10.3334/ORNLDAAC/569](https://doi.org/10.3334/ORNLDAAC/569).

## Usage of `pre_processing.py`

The `pre_processing.py` script is designed to prepare the necessary input files for the CAETÊ model by processing raw climatic and soil data. Below are the steps and examples to use this script effectively.

### Prerequisites

- Ensure you have Python installed on your system.
- Install the required dependencies by running: