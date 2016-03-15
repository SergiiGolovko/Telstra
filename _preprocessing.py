# import pandas, Series, DataFrame
from __future__ import division
import pandas as pd
import numpy as np
from pandas import DataFrame

# import time
import time

MIN_FREQ = 50


def create_order_features(df):
    """Creates orders features

    Parameters
    ----------
    df : data frame with two columns: id, location sorted by location

    Return
    ------
    df : data frame with columns:
        id, location
        order - relative order of each entry
        rank  - rank of entry within each location
        rel_rank - rank of each entry within each location
        loc_count - number of observations for specific location"""

    print "CREATING ORDER FEATURES"

    start_time = time.time()

    # transform location feature to be numerical
    # ex. "location 118" is replaces by "118"
    df["location"] = df["location"].apply(lambda val: int(val.split()[1]))

    # creating order and rank features
    df['order'] = np.arange(len(df)) / len(df)
    df['rank'] = df.groupby('location')['order'].rank().values

    # creating loc_count feature
    gdf = df[["id", "location"]].groupby("location", as_index=False).count()
    gdf.rename(columns={'id': 'loc_count'}, inplace=True)
    df['loc_count'] = pd.merge(df, gdf, how='inner', on='location')['loc_count'].values

    # creating relative rank feature
    df['rel_rank'] = df['rank'] / df['loc_count']

    elapsed_time = time.time() - start_time
    print "ALL FEATURES WERE SUCCESFULLY CREATED. TIME ELAPSED " + str(elapsed_time) + "sec."

    return df


def remove_rare_features(df, prfx):
    """Removes all features with frequencies less than MIN_FREQ,
       creates additional feature "count of rare features"

    Parameters
    ----------
    df : data frame with n features
    prfx : prefix of additional feature

    Return
    ------
    df : data frame with n-m+I(m>0) features,
         where m is the number of rare features.
         I(m>0) = 1 if there is at least one missing feature;
         then additional feature "count of rare features" is created
         with a name #rare_+prfx
    """

    print "NUMBER OF FEATURES BEFORE REMOVING RARE FEATURES %d" % df.shape[1]

    # column list
    features_list = df.columns.tolist()

    # remove a column "id"
    features_list.remove('id')

    # count of rear features for each id
    count_rare_features = np.zeros(len(df))

    for ftr in features_list:
        is_zero = 1 * (df[ftr] > 0)
        if sum(is_zero) < MIN_FREQ:
            count_rare_features += is_zero
            df.drop([ftr], axis=1, inplace=True)

    if sum(count_rare_features) > 0:
        df['#rare_' + prfx] = count_rare_features

    print "NUMBER OF FEATURES AFTER REMOVING RARE FEATURES %d" % df.shape[1]

    return df


def create_dummy_features(ids, event_df, resource_df, severity_df):
    """Creates dummy variables for event, resource and severity data frames

    Parameters
    ----------
    ids : data frame with id column
    event_df : data frame with event_type column
    resource_df : data frame with resource_type column
    severity_df : data frame with severity_type column

    Return
    ------
    df : data frame with n+m*k dummy columns,
         where n - number of event types
               m - number of resource types
               k - number of severity types
    """

    print "CREATING DUMMY FEATURES"
    start_time = time.time()

    all_df = DataFrame(ids, columns=['id'])

    dfs = [event_df, resource_df, severity_df]
    names = ['event_type', 'resource_type', 'severity_type']
    prfxes = ['event', 'resource', 'severity']

    for (df, name, prefix) in zip(dfs, names, prfxes):

        df[name] = df[name].apply(lambda val: int(val.split()[1]))

        # check that there is no duplicates in data frame, should be replaced by assert
        if not (len(df) == len(df.drop_duplicates())):
            print "There are duplicates in data frame" + prefix

        # create dummy variable for each type of event, resource or severity
        df = pd.get_dummies(df, columns=[name], prefix=prefix)

        # group by id and apply max to reduce data frame to number of ids
        df = df.groupby('id', as_index=False).max()

        # remove rare features from data frame
        df = remove_rare_features(df, prefix)

        # merge with all_df
        all_df = pd.merge(all_df, df, how='inner', on='id')

        # check that there is no duplicates in data frame, should be replaced by assert
        if not (len(all_df) == len(df)):
            print "Error: sizes of all_df and df are not the same"

    elapsed_time = time.time() - start_time;
    print "ALL FEATURES WERE SUCCESFULLY CREATED. TIME ELAPSED " + str(elapsed_time) + "sec."

    return all_df


def create_log_features(ids, feature_df):
    """Creates log features

    Parameters
    ----------
    ids : data frame with id column
    feature_df : data frame with columns
                 id - id
                 log_feature - name of feature,
                 volume - volume
    Return
    ------
    df : data frame with columns
         id - id
         name of feature - volume
    """

    print "CREATING LOG FEATURES"
    start_time = time.time()

    all_df = DataFrame(ids, columns=['id'])

    feature_df = feature_df.pivot_table(values='volume', index='id', columns='log_feature').fillna(0)
    feature_df.reset_index(level=0, inplace=True)

    if not (len(feature_df) == len(all_df)):
        print "ERROR: feature and all df frames have different sizes"

    # remove rare features from data frame
    feature_df = remove_rare_features(feature_df, 'log_feature')

    # merge with all_df
    all_df = pd.merge(all_df, feature_df, how='inner', on='id')

    elapsed_time = time.time() - start_time;
    print "ALL FEATURES WERE SUCCESSFULLY CREATED. TIME ELAPSED " + str(elapsed_time) + "sec."

    return all_df


def create_log_features2(ids, feature_df):
    """Creates log features2

    Parameters
    ----------
    ids : data frame with id column
    feature_df : data frame with columns
                 id - id
                 log_feature - name of feature in the form "log_feature space feature's number"
                               ex. log_feature 56
                 volume - volume
    Return
    ------
    df : data frame with columns
         id - id
         max_log_feature - maximum feature's number
         min_log_feature - minimum feature's number
         median_log_feature - median feature's number
         count_log_feature - number of different features
         max_volume - maximum volume
         min_volume - minimum volume
         median - median volume
         count_volume - the same as count_log_features, consider dropping!
    """

    print "CREATING LOG FEATURES"
    start_time = time.time()

    all_df = DataFrame(ids, columns=['id'])

    feature_df['log_feature'] = feature_df['log_feature'].apply(lambda val: int(val.split()[1]))

    names = ('log_feature', 'volume')

    for name in names:
        gdf = feature_df[['id', name]].groupby('id', as_index=False)

        max_feature = gdf.max()
        max_feature.columns = ['id', 'max_' + name]
        all_df = pd.merge(all_df, max_feature, how='inner', on='id')

        min_feature = gdf.min()
        min_feature.columns = ['id', 'min_' + name]
        all_df = pd.merge(all_df, min_feature, how='inner', on='id')

        median_feature = gdf.median()
        median_feature.columns = ['id', 'median_' + name]
        all_df = pd.merge(all_df, median_feature, how='inner', on='id')

        count_feature = gdf.count()
        count_feature.columns = ['id', 'count_' + name]
        all_df = pd.merge(all_df, count_feature, how='inner', on='id')

        # check whether there are null entries
    if sum(sum(1 * all_df.isnull().values)) > 0:
        print "ERROR: there are null entries in the all df data frame"

    elapsed_time = time.time() - start_time;
    print "ALL FEATURES WERE SUCCESFULLY CREATED. TIME ELAPSED " + str(elapsed_time) + "sec."

    return all_df


def create_lag_features(df, n_lags=10):
    """Creates lag features for all features in data frame except for
       id, location and order - which are necessary features

    Parameters
    ----------
    df : data frame with columns
         id, location, order and [... - referred as others]
    n_lags : number of lag features to be created

    Return
    ------
    df : data frame with columns
         id - id
         others - featurelag_i - value of feature in location corresponding to id in period t-i for i = 1:n_lags
                  featurefrwd_i - value of feature in location corresponding to id in period t+i for i = 1:n_lags
    """

    print "CREATING LAG FEATURES"

    start_time = time.time()

    # sort data frame by order
    df.sort_values(by='order', inplace=True)

    # column list
    features_list = df.columns.tolist()

    # list of features for which lag_features are created
    # remove columns 'id', 'order', 'location'
    features_list.remove('id')
    features_list.remove('order')
    features_list.remove('location')

    location = df['location'].values

    for i in range(1, n_lags + 1):
        location_frwrd_mask, location_back_mask = np.zeros(len(df)), np.zeros(len(df))
        location_mask = 1 * (location[:-i] == location[i:])
        location_frwrd_mask[:-i] = location_mask
        location_back_mask[i:] = location_mask

        for ftr in features_list:
            df[ftr + "lag_%d" % i] = location_back_mask * df[ftr].shift(1)
            df[ftr + "frwrd_%d" % i] = location_frwrd_mask * df[ftr].shift(-1)

            # fill in nonan values
    df.fillna(0, inplace=True)
    df.drop(['location', 'order'], axis=1, inplace=True)
    df.drop(features_list, axis=1, inplace=True)

    elapsed_time = time.time() - start_time;
    print "ALL FEATURES WERE SUCCESFULLY CREATED. TIME ELAPSED " + str(elapsed_time) + "sec."

    return df


def preprocessing():
    print "################PREPROCESSING - IT MAY TAKE SOME TIME - ####################"

    # read all files
    train_df, test_df = pd.read_csv('input/train.csv'), pd.read_csv('input/test.csv')
    event_df, feature_df = pd.read_csv('input/event_type.csv'), pd.read_csv('input/log_feature.csv')
    resource_df, severity_df = pd.read_csv('input/resource_type.csv'), pd.read_csv('input/severity_type.csv')

    # concatenate train and test data frames
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    print "all_df size is %d" % len(all_df)
    print "event_df size is %d and feature_df size is %d" % (len(event_df), len(feature_df))
    print "resource_df size is %d and severity_df size is %d" % (len(resource_df), len(severity_df))

    # merge severity with all_df
    # note that all_df data frame is sorted by locations
    all_df = pd.merge(severity_df[['id']], all_df, how='left', on='id')

    # create order features
    order_features = create_order_features(all_df[['id', 'location']])
    all_df = pd.merge(all_df[['id', 'fault_severity']], order_features, how='left', on='id')

    # create dummy features
    dummy_features = create_dummy_features(all_df['id'], event_df, resource_df, severity_df)
    all_df = pd.merge(all_df, dummy_features, how='left', on='id')

    # create log features
    log_features = create_log_features(all_df['id'], feature_df)
    all_df = pd.merge(all_df, log_features, how='left', on='id')

    # create log features 2
    log_features = create_log_features2(all_df['id'], feature_df)
    all_df = pd.merge(all_df, log_features, how='left', on='id')

    # create lag features for fault severity
    lag_fault_severity = create_lag_features(all_df[["id", "location", "order", "fault_severity"]], 2)
    all_df = pd.merge(all_df, lag_fault_severity, how='left', on='id')

    # create lag features for other variables
    dummy_features_set = set(dummy_features.columns.tolist())
    log_features_set = set(log_features.columns.tolist())
    all_features_set = dummy_features_set.union(log_features_set)
    all_features_set = all_features_set.union(['location', 'order'])

    lag_features = create_lag_features(all_df[list(all_features_set)], 2)
    all_df = pd.merge(all_df, lag_features, how='left', on='id')

    # split all_df data frame back into training and testing set
    train_df = all_df[~all_df['fault_severity'].isnull()]
    test_df = all_df[all_df['fault_severity'].isnull()]

    train_df.to_csv('input/train_clean.csv', index=False)
    test_df.to_csv('input/test_clean.csv', index=False)

    print "################PREPROCESSING - END - ######################################"

if __name__ == '__main__':
    preprocessing()
