## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name wlns_env python=3.7
source activate alpw_env
conda install --file requirements.txt -c pytorch
```

## Datasets
Once the datasets are downloaded, add them to the package data folder by running :
```
python ALPW/process.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

* In order to reproduce the results of ``ALPW" on the three datasets in the paper, go to the ALPW/ folder and run the following commands

```
python learner.py --dataset ICEWS14 --model TComplEx --rank 2000 --emb_reg 0.0025 --time_reg 0.001 --alpha 0.3 --beta -5

python learner.py --dataset ICEWS05-15 --model TComplEx --rank 2000 --emb_reg 0.0025 --time_reg 0.1 --alpha 0.1 --beta -1

python learner.py --dataset GDELT --model TComplEx --rank 2000 --emb_reg 0 --time_reg 0.025
```
* Results will be printed out and stored in the corresponding dataset folders.

## Acknowledgement
We refer to the code of TComplEx. Thanks for their great contributions!

## License
ALPW is CC-BY-NC licensed, as found in the LICENSE file.

