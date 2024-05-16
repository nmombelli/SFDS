import logging
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def split_data(
        dtf_in: pd.DataFrame,
        str_tgt: str = 'TARGET',
        split_strategy: str = None,
        n_splits: int = 5,
        random_state: int = None
):
    """
    Split the input data in train and test set according to the strategy desired.
    From the train set, folds for cross validation to use in the model are also retrieved.
    :param dtf_in: the dataset to split
    :param str_tgt: name of the target variable
    :param split_strategy: how to split the dataset in train and test set. Allowed values are None and OVERSAMPLING.
    :param n_splits: number of folds used for cross validation.
    :param random_state: seed to be set for reproducibility
    :return: train and test set, dictionary of the indexes for the folds used in cross validation
    """

    if split_strategy not in [None, 'OVERSAMPLING']:
        raise ValueError("split_strategy not in [None, 'OVERSAMPLING']")

    # SPLITTING
    X = dtf_in[[c for c in dtf_in if c != str_tgt]].copy()
    y = dtf_in[str_tgt].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)

    logging.debug(f"SPLITTING DATA - SIZE: TRAIN={X_train.shape[0]} - TEST={X_test.shape[0]}")
    logging.info(f"SPLITTING_DATA - TARGET TRAIN: {round(y_train.value_counts(normalize=True)[1], 6)} %")
    logging.info(f"SPLITTING_DATA - TARGET TEST:  {round(y_test.value_counts(normalize=True)[1], 6)} %")

    if split_strategy == 'OVERSAMPLING':
        sm = SMOTE(sampling_strategy=0.3, random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logging.info(f"SPLITTING_DATA - TARGET TRAIN SMOTE: {round(y_train.value_counts(normalize=True)[1], 6)} %")
        # the shape of the train changes, the shape of the test not.
        logging.debug(f"SPLITTING DATA - SIZE: TRAIN={X_train.shape[0]}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dct_cv = {}

    for i, (idx_train, idx_val) in enumerate(skf.split(X=X_train, y=y_train)):
        dct_cv[i] = {}
        dct_cv[i]['TRAIN'] = idx_train
        dct_cv[i]['VALIDATION'] = idx_val
        logging.debug(f"SPLITTING_DATA - FOLD: {i}: {round(y_train.iloc[idx_val].value_counts(normalize=True)[1], 6)}%")

    # verify schema
    if set(dtf_in) != set(X_train).union({'TARGET'}):
        raise ValueError('TRAIN columns not as expected')
    if set(dtf_in) != set(X_test).union({'TARGET'}):
        raise ValueError('TEST columns not as expected')

    return X_train, X_test, y_train, y_test, dct_cv
