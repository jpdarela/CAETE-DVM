
import sys
import numpy as np
import polars as pl
import altair as alt


sys.path.append("../src/")
import _geos as geos
from config import fetch_config


cfg = fetch_config()

xres, yres= cfg.crs.xres, cfg.crs.yres
xlen = int(360 // xres)
ylen = int(180 // yres)


# bounding box indices in the (ylen, xlen) grid
bb_idx = type("PanAmazon", (), geos.pan_amazon_region)()

# read model output data
df = pl.read_parquet("../outputs/pan_amazon_hist_biomass_2021.parquet")

# Read PLS_data
pls_df = pl.read_csv("../outputs/pls_attrs-200000.csv")


def weighted_mean(df, value_col, weight_col):
    weighted_sum = (df[value_col] * df[weight_col]).sum()
    total_weight = df[weight_col].sum()
    return weighted_sum / total_weight


def biomass_cwm(df, value_col, weight_col:str = "ocp"):
    # Select all gridcells iwith the same latitude and longitude
    # weighted mean cveg poools by ocp 
    out = np.zeros((ylen, xlen), dtype=np.float64) + 1.0e20
    for y in range(bb_idx.ymin, bb_idx.ymax):
        for x in range(bb_idx.xmin, bb_idx.xmax):
            filt = df.filter(
                (pl.col("grid_y") == y) &
                (pl.col("grid_x") == x))
            
            if filt.height == 0:
                continue
            
            wm_cveg = weighted_mean(filt, value_col= value_col, weight_col= weight_col)
            out[y, x] = wm_cveg

    out = np.ma.masked_array(out, mask= out == 1.0e20)

    # CLip to pan-amazon region
    return out[bb_idx.ymin:bb_idx.ymax, bb_idx.xmin:bb_idx.xmax]


def trait_cwm0(biomass_df, pls_df, trait_col, weight_col: str = "ocp"):
    # Select all gridcells iwith the same latitude and longitude
    # weighted mean of traits in the pls_table by ocp 
    out = np.zeros((ylen, xlen), dtype=np.float64) + 1.0e20
    
    for y in range(bb_idx.ymin, bb_idx.ymax):
        for x in range(bb_idx.xmin, bb_idx.xmax):
            filt = biomass_df.filter(
                (pl.col("grid_y") == y) &
                (pl.col("grid_x") == x))
            
            nrows = filt.height
            if nrows == 0:
                continue

            pls_ids = filt.select(pl.col("pls_id")).to_series()
            weights = filt.select(pl.col(weight_col)).to_series()

            # Get trait values for the PLS ids present in this gridcell
            trait_values = []
            for pls_id in pls_ids:
                trait_row = pls_df.filter(pl.col("PLS_id") == pls_id)
                if trait_row.height > 0:
                    trait_val = trait_row.select(pl.col(trait_col)).item()
                    trait_values.append(trait_val)
                else:
                    # Handle missing PLS_id - you might want to skip or use a default value
                    trait_values.append(np.nan)
            
            trait_values = np.array(trait_values)
            weights_array = weights.to_numpy()
            
            # Calculate weighted mean, excluding NaN values
            valid_mask = ~np.isnan(trait_values)
            if valid_mask.sum() > 0:
                wm_trait = np.average(trait_values[valid_mask], weights=weights_array[valid_mask])
                out[y, x] = wm_trait

    out = np.ma.masked_array(out, mask= out == 1.0e20)

    # Clip to pan-amazon region
    return out[bb_idx.ymin:bb_idx.ymax, bb_idx.xmin:bb_idx.xmax]


def trait_cwm(biomass_df, pls_df, trait_col, weight_col: str = "ocp"):
    # Create a lookup dictionary for traits (much faster than repeated filtering)
    trait_lookup = dict(zip(pls_df["PLS_id"], pls_df[trait_col]))
    
    # Group by grid coordinates to avoid repeated filtering
    grouped = biomass_df.group_by(["grid_y", "grid_x"]).agg([
        pl.col("pls_id"),
        pl.col(weight_col)
    ])
    
    out = np.zeros((ylen, xlen), dtype=np.float64) + 1.0e20
    
    # Process each group
    for row in grouped.iter_rows(named=True):
        y, x = row["grid_y"], row["grid_x"]
        
        # Skip if outside bounding box
        if not (bb_idx.ymin <= y < bb_idx.ymax and bb_idx.xmin <= x < bb_idx.xmax):
            continue
            
        pls_ids = row["pls_id"]
        weights = row[weight_col]
        
        # Get trait values using dictionary lookup
        trait_values = np.array([trait_lookup.get(pls_id, np.nan) for pls_id in pls_ids])
        weights_array = np.array(weights)
        
        # Calculate weighted mean, excluding NaN values
        valid_mask = ~np.isnan(trait_values)
        if valid_mask.sum() > 0:
            wm_trait = np.average(trait_values[valid_mask], weights=weights_array[valid_mask])
            out[y, x] = wm_trait

    out = np.ma.masked_array(out, mask=out == 1.0e20)
    return out[bb_idx.ymin:bb_idx.ymax, bb_idx.xmin:bb_idx.xmax]
if __name__ == "__main__":
    pass
    # cveg_wmean = biomass_cwm(df, value_col="cveg", weight_col="ocp")
    # leafp_wme = biomass_cwm(df, value_col="vp_cleaf", weight_col="ocp")
    # frootp_wme = biomass_cwm(df, value_col="vp_croot", weight_col="ocp")
    # # np.save("../outputs/pan_amazon_cveg_weighted_mean.npy", cveg_wmean)

# Select one gridcell (163, 236) and year 1901
# filt = df.filter(
#     (pl.col("year") == 1901) &
#     (pl.col("grid_y") == 163) &
#     (pl.col("grid_x") == 236))

# filt = df.filter(
#     (pl.col("year") == 1901))

# chart = alt.Chart(filt).mark_line().encode(
#     x='ocp',
#     y='cveg',
#     tooltip=['ocp', 'cveg']
# )
# # chart.show()
