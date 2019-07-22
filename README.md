# FuseLSTM_AA
This package uses LSTM to do time-series analysis and estimate crack length (or variable of interest) based on features of data corresponding to a particular time window prior to the time of interest. 

To use this package, follow the steps listed below.

1- To do a grid search at first, run "BiLSTM_AA_GS". Determin grid search intervals, the input file, columns (features) to be considered, and the label column 
(e.g., size column in the case of crack size estimation)

2- Considering the output of the grid search, find the best model and use "BiLSTM_AA.py" to train that model. Determine input variables and file in the 
"Input variables/paramters" section. 

3- Use "BiLSTM_AA_predict.py" to predict the value of interest. The output will be a file including input features (or size estimates) and the predicted value 
(e.g., final damage size estimate)
