
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
b2000 = pl.read_parquet("../outputs/pan_amazon_hist_biomass_1961.parquet")
b0 = pl.read_parquet("../outputs/pan_amazon_hist_biomass_1901.parquet")

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


# def trait_cwm0(biomass_df, pls_df, trait_col, weight_col: str = "ocp"):
#     # Select all gridcells iwith the same latitude and longitude
#     # weighted mean of traits in the pls_table by ocp 
#     out = np.zeros((ylen, xlen), dtype=np.float64) + 1.0e20
    
#     for y in range(bb_idx.ymin, bb_idx.ymax):
#         for x in range(bb_idx.xmin, bb_idx.xmax):
#             filt = biomass_df.filter(
#                 (pl.col("grid_y") == y) &
#                 (pl.col("grid_x") == x))
            
#             nrows = filt.height
#             if nrows == 0:
#                 continue

#             pls_ids = filt.select(pl.col("pls_id")).to_series()
#             weights = filt.select(pl.col(weight_col)).to_series()

#             # Get trait values for the PLS ids present in this gridcell
#             trait_values = []
#             for pls_id in pls_ids:
#                 trait_row = pls_df.filter(pl.col("PLS_id") == pls_id)
#                 if trait_row.height > 0:
#                     trait_val = trait_row.select(pl.col(trait_col)).item()
#                     trait_values.append(trait_val)
#                 else:
#                     # Handle missing PLS_id - you might want to skip or use a default value
#                     trait_values.append(np.nan)
            
#             trait_values = np.array(trait_values)
#             weights_array = weights.to_numpy()
            
#             # Calculate weighted mean, excluding NaN values
#             valid_mask = ~np.isnan(trait_values)
#             if valid_mask.sum() > 0:
#                 wm_trait = np.average(trait_values[valid_mask], weights=weights_array[valid_mask])
#                 out[y, x] = wm_trait

#     out = np.ma.masked_array(out, mask= out == 1.0e20)

#     # Clip to pan-amazon region
#     return out[bb_idx.ymin:bb_idx.ymax, bb_idx.xmin:bb_idx.xmax]


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
    b2021 = biomass_cwm(df, "cveg")
    b2001 = biomass_cwm(b2000, "cveg")
    b1901 = biomass_cwm(b0, "cveg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    im0 = ax[0].imshow(b1901, cmap="viridis")
    ax[0].set_title("Biomass 1901")
    fig.colorbar(im0, ax=ax[0], label="Biomass (kg C m⁻²)")

    im1 = ax[1].imshow(b2001, cmap="viridis")
    ax[1].set_title("Biomass 2001")
    fig.colorbar(im1, ax=ax[1], label="Biomass (kg C m⁻²)")

    im2 = ax[2].imshow(b2021, cmap="viridis")
    ax[2].set_title("Biomass 2021")
    fig.colorbar(im2, ax=ax[2], label="Biomass (kg C m⁻²)")

    diff_21_1901 = b2021 - b1901
    v1 = np.ma.max(np.abs(diff_21_1901))
    im3 = ax[3].imshow(diff_21_1901, cmap="bwr", vmin=-v1, vmax=v1)
    ax[3].set_title("Biomass Change (2021 - 1901)")
    fig.colorbar(im3, ax=ax[3], label="Biomass Change (kg C m⁻²)")

    diff_21_2001 = b2021 - b2001
    v2 = np.ma.max(np.abs(diff_21_2001))
    im4 = ax[4].imshow(diff_21_2001, cmap="bwr", vmin=-v2, vmax=v2)
    ax[4].set_title("Biomass Change (2021 - 1961)")
    fig.colorbar(im4, ax=ax[4], label="Biomass Change (kg C m⁻²)")

    # unused subplot
    ax[5].axis("off")

    plt.tight_layout()
    plt.show()

    aroot2021 = trait_cwm(df, pls_df, trait_col="aroot")
    aroot2001 = trait_cwm(b2000, pls_df, trait_col="aroot")
    aroot1901 = trait_cwm(b0, pls_df, trait_col="aroot")

    diff_21_1901 = aroot2021 - aroot1901
    diff_21_2001 = aroot2021 - aroot2001

    plt.figure(figsize=(12, 5))
    v = np.ma.max(np.abs(diff_21_1901))
    im = plt.imshow(diff_21_1901, cmap="bwr", vmin=-v, vmax=v)
    plt.title("Root allocation Change (2021 - 1901)")

    plt.colorbar(im, label="Root Allocation Change")
    plt.show()
    plt.figure(figsize=(12, 5))
    v = np.ma.max(np.abs(diff_21_2001))
    im = plt.imshow(diff_21_2001, cmap="bwr", vmin=-v, vmax=v)
    plt.title("Root allocation Change (2021 - 1961)")
    plt.colorbar(im, label="Root Allocation Change")
    plt.show()
    


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
