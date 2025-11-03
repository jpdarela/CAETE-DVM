# Input Data

[⇦ Back](./README.md)

Input data is sourced from the **ISIMIP repository**.

More information about the input data structure and preprocessing scripts can be found [here](./input/README.md).

This repository contains a folder named `input` which includes:

- `preprocess_caete.py`: Reads ISIMIP netCDF files and creates netCDF input files for CAETÊ.
- `preprocess_caete_pbz2.py`: Converts ISIMIP netCDF files into legacy `.pbz2` input files.
- The new input handler supports both formats, configurable via `caete.toml`.

The model executes simulations based on a **gridlist** table. Preprocessing scripts are designed to save these gridlists alongside the input files.

In `input_handler.py`, the `input_handler` class includes a static method that can generate a gridlist from a properly formatted netCDF file. This functionality is also integrated into the `bz2_handler` module, although it has not been used recently.

### Input Sources

- **Protocols**: ISIMIP3a and ISIMIP3b
- **Bias-corrected data**: Stored in folders with a `_raw` suffix (downloaded from ISIMIP).
- **Input folders**: `spinclim`, `obsclim`, `historical`, etc.

### Download Links

- **20CRv3-ERA5** (`spinclim` & `obsclim`): [Download](https://1drv.ms/f/c/16d3a2cfff38aeca/EsquOP_PotMggBaPxwMAAAABjdx1IH1OuwQ8THwZcD8izw?e=lDgnH4)
- **MPI-ESM1-2-HR** (`piControl`, `historical`, `ssp370`, `ssp585`): [Download](https://1drv.ms/f/c/16d3a2cfff38aeca/EsquOP_PotMggBaOHwQAAAABcpwkuYUk34IHwz9sxJIWnw?e=AGnyYC)
- **Gridlists**: [`CAETE-DVM/grd`](./grd/)

---
[⇦ Back](./README.md)