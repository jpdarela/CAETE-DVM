# Input data for CAETÊ

This folder contains the input data used to run the CAETE model. These data are divided into
folders:

[mask]("./mask/"), [co2]("./co2/"), [hydra]("./hydra/"), and [soil]("./soil/") contains some boolean masks used to preprocess input data and configure model execution. There are also files with soil hydraulic parameters and nutrient content (N & P). The co2 folder has timeseries of annual ATM CO2 concentration. Observed and projected.

[20CRv3-ERA5]("./20CRv3-ERA5/") and [MPI-ESM1-2-HR](./MPI-ESM1-2-HR/) store climatic drivers. Both folders should contain data downloaded from the ISIMIP data repository. The netCDF files with climatic variables (e.g., tas, hurs) must be organized into a subfolder, like in the following way:

```./MPI-ESM1-2-HR/ssp585_raw```

This folder contains all the netCDF files for all needed variables:

```hurs, tas, pr, ps, rsds, sfcwind```

You can find the reference to all datasets used here at the end of this document.

In the example above,

```./MPI-ESM1-2-HR/ssp585_raw```

the netCDF files downloaded from the ISIMIP repository must be placed into a folder that has the identification of the dataset, in this case ```ssp585``` appended with a string ```_raw```, making ```ssp585_raw``` indicating that these are the files that we need to process in order to have files that CAETE can read.

If these assumptions are met then you can use the pre_processing.py script to prepare files that are employed to feed the CAETÊ model. The raw climatic and edaphic data in these files are publicly available from other sources.

For example, to process the netCDF files in the example above you can run:

```$ python pre_processing.py --dataset MPI-ESM1-2-HR --mode ssp585```

* Note that you use only the identification of the dataset: ```ssp585``` for the flag ```--mode```. The first flag, ```--dataset``` indicates the climatic dataset that is used: ```MPI-ESM1-2-HR```.

This program will create a folder called ```ssp585``` like:

```./MPI-ESM1-2-HR/ssp585/```

This folder contains the data in another format. Each file for a different gridcell. Each file contains also data on soil nutrients and hydraulic parameters needed by the model in order to execute the processes for a given gridcell.

Climatic data: ISIMIP3a/b

Soil data: HWSD, IGBP, & Darela-Filho et al., 2024

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