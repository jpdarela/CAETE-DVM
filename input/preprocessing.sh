pyenv local 3.11.7

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python pre_processing.py --dataset MPI-ESM1-2-HR --mode  historical
# python pre_processing.py --dataset MPI-ESM1-2-HR --mode  piControl
python pre_processing.py --dataset MPI-ESM1-2-HR --mode  ssp370
python pre_processing.py --dataset MPI-ESM1-2-HR --mode  ssp585