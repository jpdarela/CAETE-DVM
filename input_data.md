# Input Data

[⇦ Back](./README.md)

Input data were downloaded from the **ISIMIP repository**.

More information about the input data structure and preprocessing scripts can be found [here](./input/README.md).


In this repository there is a folder called `input` containing:

- `preprocess_caete.py`: Reads ISIMIP netCDFs and creates netCDF input files for CAETE.
- `preprocess_caete_pbz2.py`: Converts ISIMIP netCDFs into legacy `.pbz2` input files.
- The new input handler supports both formats (configured via `caete.toml`).

The execution of gridcells depends on a **gridlist** table. Preprocessing scripts should save gridlists alongside input files.

In `input_handler.py`, the class `input_handler` includes a static method that can generate a gridlist from a properly formatted netCDF file (but not necessarily from a folder of `.pbz2` files). This feature was also integrated into the `bz2_handler` module, although it has not been used recently.

### Input Sources

- **Protocols**: ISIMIP3a and ISIMIP3b
- **Bias-corrected data**: Stored in folders with `_raw` suffix (downloaded from ISIMIP)
- **Input folders**: `spinclim`, `obsclim`, `historical`, etc.

### Download Links

- **20CRv3-ERA5** (`spinclim` & `obsclim`): [Download](https://1drv.ms/f/c/16d3a2cfff38aeca/EsquOP_PotMggBaPxwMAAAABjdx1IH1OuwQ8THwZcD8izw?e=lDgnH4)
- **MPI-ESM1-2-HR** (`piControl`, `historical`, `ssp370`, `ssp585`): [Download](https://1drv.ms/f/c/16d3a2cfff38aeca/EsquOP_PotMggBaOHwQAAAABcpwkuYUk34IHwz9sxJIWnw?e=AGnyYC)
- **Grid lists**: [`CAETE-DVM/grd`](./grd/)

---
[⇦ Back](./README.md)