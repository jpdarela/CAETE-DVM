# Climatic input data preparation for CAETÊ

## SAVE the climate data in this folder

You can download the historical data for the variables tas, pr, ps, rsds & hurs for the time interval 1971-2010 [here](https://1drv.ms/u/s!AsquOP_PotMWgeM-nhGf3GkxV1Wq0g?e=525apd).

Save the compressed files into this folder and decompress it using tar:

`$ tar -xvzf ISIMIP-tas.tar.gz`

Do it for all arquives. You can also use another program to extract the climate files here.

You will need the MPI library and the python module mpi4py to be installed in your system.

With all files in place run:

`$ mpiexec -n 5 python3.x create_input.py`

This will create five files named:
`ps.pkl`
`pr.pkl`
`rsds.pkl`
`tas.pkl`
`hurs.pkl`

Look the python sources in this folder to modify the input creation.
