# KDDCup2021
MSBD5002 Data Mining course project

--------
## Required packages
- Numpy, Pandas, Scipy
- rrcf for robust random cut forest
- stumpy for matrix profile
- sranodec for spectral residual

---------

We ensemble several methods for anomaly detection, including statistical methods (`statistic_func.py`), fourier transformation （`fourier_transform.py`）, matrix profile (`matrix_profile.py`), spectural residual (`spectral_residual.py`) and robust random cut forest (`rrcf.py`). Details of each method cound be found in each of the python file.

To run the program, use the following command 
```
bash ./run.sh
```

It may take a few days due to the significant resources required to compute the matrix profile, although we use a 6-core GPU. 

The ensembled results are saved in ensemble_results folder. We take the index with the largest confidence score to be the final detected anomaly location. The final result is in submission.csv
