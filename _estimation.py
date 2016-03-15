# import pandas, DataFrame
from __future__ import division
import pandas as pd
from pandas import DataFrame

# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# import sklearn
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# import time
import time

CURR_FIGURE = 1


def tune_parameters(estimator, name, parameters_grid, train_X, train_y):
    """Returns the best set of parameters out of ones specified in parameters_grid

    Parameters
    ----------
    estimator : estimator for which parameters are tuned
    name : name of estimator
    parameters_grid : dictionary or list of dictionaries with parameters to be tuned
    train_X : data frame with features
    train_y : data frame with labels

    Return
    ------
    best_parameter : dictionary of best parameters
    best_score : float
    """

    print "Tuning parameters for " + name

    start_time = time.time()

    cv = KFold(n=len(train_y), n_folds=2, shuffle=True, random_state=1)

    gscv = GridSearchCV(estimator,
                        parameters_grid,
                        cv=cv,
                        scoring='log_loss',
                        verbose=100)
    gscv.fit(train_X, train_y)

    print "The mean score and all cv scores are"
    for params, mean_score, cv_scores in gscv.grid_scores_:
        print("%0.3f %s for %r" % (mean_score, np.array_str(cv_scores, precision=3), params))

    elapsed_time = time.time() - start_time

    print "The best score is %0.3f and the best parameters are %r" %(gscv.best_score_, gscv.best_params_)
    print "Finished tuning parameters, time elapsed " + str(elapsed_time) + "sec."

    return [gscv.best_params_, gscv.best_score_]


def plot_gb_performance(gb, train_X, train_y):
    """Plots the performance of gradient boosting as a function of number of trees

    Parameters
    ----------
    gb : gradient boosting estimator
    train_X : data frame with features
    train_y : data frame with labels

    Returns
    -------
    The plot of gradient boosting performance
    """

    global CURR_FIGURE

    print "Plotting gradient boosting performance"

    # split training set into training set and validation set
    train_X, val_X, train_y, val_y = train_test_split(train_X, 
                                                      train_y, 
                                                      test_size=0.5, 
                                                      random_state=1)
                                                      
    gb.fit(train_X, train_y)
    
    
    # this part of the code is adapted from scikit learn documentation
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    # compute test set mse
    n_estimators = gb.get_params()['n_estimators']
    test_score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(gb.staged_predict_proba(val_X)):
        test_score[i] = log_loss(y_true=val_y, y_pred=y_pred)

    plt.figure(CURR_FIGURE)
    plt.title('Gradient Boosting Performance')
    plt.plot(np.arange(n_estimators) + 1, gb.train_score_/len(train_X), 'b-', label='Training Set')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-', label='Test Set')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Score')
    CURR_FIGURE += 1


def plot_rf_performance(rf, train_X, train_y, at_least=100):
    """Plots the performance of random forest as a function of number of trees

    Parameters
    ----------
    rf : random forest estimator
    train_X : data frame with features
    train_y : data frame with labels
    at_least : float, plot the performance starting with at least at_least trees

    Return
    ------
    Plot of random forest performance
    """

    global CURR_FIGURE

    print "Plotting random forest performance"

    #split training set into training set and validation set
    train_X, val_X, train_y, val_y = train_test_split(train_X, 
                                                      train_y, 
                                                      test_size=0.5, 
                                                      random_state=0)
    
    rf.fit(train_X, train_y)

    # this part of the code is adapted from scikit learn documentation
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    # compute test set log loss score
    n_estimators = rf.get_params()['n_estimators']

    # empty arrays for train and test scores
    n_classes = rf.n_classes_
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    train_score = np.zeros((n_estimators,), dtype=np.float64)
    
    for scores, y, X in zip( [train_score, test_score], [train_y, val_y], [train_X, val_X] ):

        # aggregate pred y for all trees up to current tree
        aggr_y = np.zeros((y.shape[0], n_classes), dtype=np.float64)
        
        for i, tree in enumerate(rf.estimators_):
        
            pred_y = tree.predict_proba(X)
        
            # update aggregate y
            aggr_y = (aggr_y * i + pred_y)/(i+1)
        
            scores[i] = log_loss(y_true=y, y_pred=aggr_y)

    plt.figure(CURR_FIGURE)
    plt.title('Random Forest Performance')
    plt.plot(np.arange(n_estimators - at_least) + at_least + 1, train_score[at_least:], 'b-', label='Training Set')
    plt.plot(np.arange(n_estimators - at_least) + at_least + 1, test_score[at_least:], 'r-', label='Test Set')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    CURR_FIGURE += 1

    print "The end of plotting"


def my_cross_validation(estimator, X, y, n_folds=10):
    """Does cross validation

    Parameters
    ----------
    estimator : estimator for which cross validation is dome
    X : data frame with features
    y : data frame with labels
    n_folds : float, number of folds

    Return
    ------
    np array of length n_folds
    """

    # do K-fold validation manually
    kf = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=1)
    # scores
    scores = []
    
    for train_ind, test_ind in kf:
        
        # fit the model on training set
        iX, iy = X.values[train_ind], y[train_ind]
        estimator.fit(iX, iy)
        
        # make a prediction for test set
        iX, iy = X.values[test_ind], y[test_ind]
        pred_y = estimator.predict_proba(iX)
        
        # calculate the score
        score = log_loss(y_true=iy, y_pred=pred_y)
        scores.append(score)
        
    return np.array(scores)


def output_results(ids, predictions, file_name='output/output'):
    """ Writes results to file

    Parameters
    ---------
    ids : data frame of ids
    predictions : data frame of predictions
    file_name : file name

    Return
    ------
    File saved in csv format with name file_name
    first column - ids, second columns - predictions"""
    
    output_df = DataFrame(np.concatenate((ids, predictions), axis=1), columns=['id', 'predict_0', 'predict_1', 'predict_2'])
    output_df['id'] = output_df['id'].astype(int)
    output_df.to_csv(file_name + '.csv', index=False)


def plot_feature_importance(estimator, columns, n=50):
    """ Plots feature importance

    Parameters
    ----------
    estimator : estimator for which the important features needs to be drawn
    columns : names of all features
    n : int, number of important features to plot

    Return
    ------
    Plot of features importance
    """

    global CURR_FIGURE

    # extract feature importance and normalize them to sum up to 100
    feature_importance = estimator.feature_importances_
    feature_importance = (100.0 * feature_importance) / sum(feature_importance)
    index = np.argsort(feature_importance)[::-1][0:n]
    
    # feature names
    feature_names = columns
    
    # plot
    plt.figure(CURR_FIGURE)
    pos = (np.arange(n) + .5)[::-1]
    plt.barh(pos, feature_importance[index], align='center')
    plt.yticks(pos, feature_names[index])
    plt.xlabel('Relative Importance')
    plt.title(str(n) + ' Most Important Features')
    CURR_FIGURE += 1


def estimation():
    
    print "################ESTIMATION - IT MAY TAKE SOME TIME - ####################"
    
    # read all files
    train_df = pd.read_csv('input/train_clean.csv')
    test_df = pd.read_csv('input/test_clean.csv')

    # split both train and test sets into features and labels
    train_y = train_df.fault_severity
    train_df.drop(["id", "fault_severity"], axis=1, inplace=True)
    train_x = train_df
    ids = test_df["id"].apply(lambda val: int(val)).values
    ids = np.reshape(ids, (len(ids), 1))
    test_df.drop(["id", "fault_severity"], axis=1, inplace=True)
    test_x = test_df

    # 1. random forest
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_features=0.15, criterion='entropy', random_state=0)
    rf_parameter_grid = {'max_features': [0.1, 0.05, 0.2, 0.3, 0.5, 'auto', 'sqrt']}
    plot_rf_performance(rf, train_x, train_y)

    # 2. gradient boosting
    gb = GradientBoostingClassifier(n_estimators=150,
                                    subsample=0.5,
                                    max_depth=5,
                                    learning_rate=0.1,
                                    random_state=0)
    gb_parameter_grid = {'max_depth': [1, 3, 5, 6, 10]}
    plot_gb_performance(gb, train_x, train_y)

    # 3.xgboost classifier
    xgb_clf = XGBClassifier(base_score=0.5,
                            colsample_bylevel=1,
                            colsample_bytree=0.5,
                            gamma=1,
                            learning_rate=0.1,
                            max_delta_step=0,
                            max_depth=8,
                            min_child_weight=1,
                            missing=None,
                            n_estimators=100,
                            nthread=-1,
                            objective='multi:softprob',
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=0,
                            silent=True,
                            subsample=1)

    xgb_parameter_grid = {'max_depth': [2, 4, 6, 8, 10], 'n_estimators': [50, 100, 150, 200]}

    # 4. estimate rf and gb and do cross validation
    estimators, names = (xgb_clf, rf, gb), ("XGBClassifier", "Random Forest", "Gradient Boosting")
    parameters_grid = (xgb_parameter_grid, rf_parameter_grid, gb_parameter_grid)

    results = None

    for estimator, name, grid in zip(estimators, names, parameters_grid):

        param, best_score = tune_parameters(estimator, name, grid, train_x, train_y)

        start_time = time.time()

        print "Fitting %s model" % name

        estimator = estimator.set_params(**param)
        estimator.fit(train_x, train_y)
        y_pred = estimator.predict_proba(test_x)

        # temporary cross validation
        #print "Starting cross validation for %s model" % name
        #cv = KFold(n=len(train_y), n_folds=2, shuffle=True, random_state=1)
        #cv_scores = cross_val_score(estimator, train_x, train_y, cv=cv, scoring='log_loss', n_jobs=1)
        #print "My score validation %0.3f %s " % (cv_scores.mean(), np.array_str(cv_scores, precision=3))
        #print "End of cross validation"

        if results is None:
            results = y_pred
        else:
            results = np.concatenate((results, y_pred), axis=1)

        elapsed_time = time.time() - start_time
        print "The model was successfully estimated, time elapsed " + str(elapsed_time) + "sec."

    xgb_pred, rf_pred, gb_pred = results[:, 0:3], results[:, 3:6], results[:, 6:9]

    # 5. plotting feature importance for gradient boosting
    plot_feature_importance(gb, train_x.columns)
    plot_feature_importance(rf, train_x.columns)

    # 6. show all plots
    plt.show()

    # 6. output results
    output_results(ids, xgb_pred, file_name='output/xgb_output')
    output_results(ids, gb_pred, file_name='output/gb_output')
    output_results(ids, rf_pred, file_name='output/rf_output')
    output_results(ids, 0.5*xgb_pred + 0.25*gb_pred + 0.25*rf_pred)

    print "################ESTIMATION - END - ######################################"

if __name__ =='__main__':
    estimation()


