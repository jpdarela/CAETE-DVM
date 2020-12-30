# Climatic input data for CAETÊ (in prep)

## pre_processing.py

## To run in you computer
 - SAVE the sample of CAETÊ input data in the folder input/caete_input
 - EDIT the file model_driver.py so the search dir. for input data matches the above folder in you computer.

## To run in sombrero
Contact me to have access to our processor server sombrero

## Raw data and SAMPLE input data for caete
You can find a sample of input data to run CAETÊ (Pan Amazon region) [here](https://1drv.ms/u/s!AsquOP_PotMWgeNdATQs9o9NVQcK6g?e=qmCFRX). The dataset is a bunch of files containing climatic and soil data. Each file contains the climatic variables hurs, tas, ps, pr and rsds for each gridcell for the time span 1901-2016. They also store the soil data used to start the model. The files are python pickled dictionaries compressed with the bz2 algorithm from the python standard library. The files where created using python 3.8.5 in a linux machine. I recomend that you inspect the files opening it with the same version of python that you will use to run the model and checking the integrity of the data. For example, you can load a input file from your python terminal:

`>>> import _pickle as pkl
`>>> import bz2
`>>> with bz2.BZ2File("input_file-y-x.pbz2" mode='r') as fh:
.........data = pkl.load(fh)`

This will create a dictionary called data containig input data for the gridcell(y,x).

You can download a sample of the raw  global historical data in the form of netCDF4 for the variables tas, pr, ps, rsds & hurs for the time interval 1971-2010 [here](https://1drv.ms/u/s!AsquOP_PotMWgeM-nhGf3GkxV1Wq0g?e=525apd). This is the raw data and is not prepared to run the model.
