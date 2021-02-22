#!/bin/bash
echo "Calculating monthly means"
echo "N files : $#"
echo "1st argument = $1"
echo "2nd Argument = $2"
echo "3rd argument = $3"
echo "VAR = $4"

ncrcat $1 $2 $3 "$4_1979-2014_daily.nc4"
cdo -O -P 4 monmean "$4_1979-2014_daily.nc4" "$4_1979-2014d_monthly.nc4"
ncks -C -x -v time_bnds "$4_1979-2014d_monthly.nc4" "$4_1979-2014_monthly.nc4"
ncatted -a bounds,time,d,, "$4_1979-2014_monthly.nc4"
rm -rf "$4_1979-2014_daily.nc4"
rm -rf "$4_1979-2014d_monthly.nc4"

echo "DOne"
