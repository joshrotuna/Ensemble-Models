# Ensemble Method
This project uses the Ensemble method algorithms (Random Forest, AdaBoost, Gradient Boosting, XGBoost) to create predictions for individual income levels. Performance metrics are evaluated as well.

## Steps
 1. Examining descriptive stats of the entire dataset
 2. Create training and testing datasets
 3. Find Optimal Max_depth value
 4. Training models to run a specificed amount of decision trees (N_estimators):
	 - Random Forest
	 - AdaBoost
	 - Gradient Boosting 
	 - XGBoost
5. Examining performance statistics of each model

## Requirements

* Python
* Google Colab

## Packages 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from google.colab import drive

from sklearn import metrics

from sklearn.metrics import roc_auc_score

!pip install xgboost

import xgboost as xgb
## Launch

Download *Joshua_Rotuna_CA04* and open the file in Google Colab.

In Google Drive, create the file structure below:

'/content/drive/MyDrive/MSBA_Colab_2020/ML_Algorithms/CA04/census_data.csv'

Next, open the attached *census_data* file and import the file into the previously created folder. 

Finally, execute the code.

## Authors

[Joshua Rotuna](https://github.com/joshrotuna)

## License

This project is licensed under the  [MIT](https://choosealicense.com/licenses/mit/)  License.

## Acknowledgements

The project files were provided by Arin Brahma, Loyola Marymount University.
