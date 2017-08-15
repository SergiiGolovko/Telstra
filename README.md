## This repository contains two scripts: _preprocessing.py and _estimation.py. 

### Both scripts are modifications of my original solution (which produces score 0.53203 - 431st place on Private Leaderboard) after competition was over in order to implement some of the ideas discussed in the Forum. The output of these scripts is output.cv file in “output” folder in the working directory and should produce score about 0.41247 - 22nd place on Private Leaderboard). 

### Before running scripts, all data files from competition (https://www.kaggle.com/c/telstra-recruiting-network under Get the Data tab) should be placed in “input” folder in the working directory. _preprocessing.py should be run first, followed by _estimation.py.

### The final model is an ensemble of XGBClassifier (with weight 0.5), Random Forest (0.25) and GradientBoosting (0.25).

### Scripts description
* _preprocessing.py - preprocess the data by extracting features and creating new features such as: 
	* order - order of each entry in severity file 
	* rank  - rank of entry within each location (as specified by order) 
	* rel_rank - rank of each entry with each location normalized to be in [0,1] interval
	* loc_count - number of entries for each location
	* dummy features for event, resource and severity types 
	* log feature volume - volume features for log features
	* max, min, median and count of log feature number and volume
	* lag features for all features (except for log volume features) for two periods before and after fault severity
	* lag features for fault severity for two periods before and after fault severity 
* _estimation.py - estimates Random Forest, Gradient Boosting, XGBoost classifiers and saves the predictions for each model in “output” folder; saves the predictions for a final model in “output” folder; and plots the performance of Random Forest and Gradient Boosting as a function of number of trees.    

    
