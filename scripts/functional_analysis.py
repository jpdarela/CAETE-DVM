
import sys
import polars as pl
import altair as alt


sys.path.append("../src/")
import _geos as geos


df = pl.read_parquet("../outputs/pan_amazon_hist/pan_amazon_hist_biomass.parquet")


# Select one gridcell (163, 236) and year 1901
filt = df.filter(
    (pl.col("year") == 1901) &
    (pl.col("grid_y") == 163) &
    (pl.col("grid_x") == 236))

chart = alt.Chart(filt).mark_line().encode(
    x='ocp',
    y='cveg',
    tooltip=['ocp', 'cveg']
)
chart.show()
