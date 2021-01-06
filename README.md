# DU_test
Test done for the recruitment process

I've stored all the data you've been giving in the Data folder.

I've created a first .py file in order to explore quite quickly the datasets. This file is data_exploration.py and it stores its results in the Exploration folder
( notice I've made comments directly in the code about theses results). The final part of the code is about formatting the dataset for the next step which is
the random forest.

Then random_forest.py calibrates a Random Forest on the train dataset using a GridSearchCV based on the AUC-PR score (calculated via average precision).
After the calibration fo the parameters the best estimator is used to predict the probability on the test dataset. The results are stored in final.csv.
The best estimator is choosen only via the maximisation of the mean score but it can be done by using the mean score and its standard error (mean and std
evaluated thanks to the folds on the train dataset).
The perfomance of every tested parameters set is stored in calibration_results.csv and their rank is directly avalaible .
