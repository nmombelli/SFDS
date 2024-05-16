import pandas as pd


def join_output(X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    join the test set with the target and the prediction. Then sort the instances by score and assign a rank.
    :param X_test: test set.
    :param y_test: target variable in the test set.
    :param y_pred: predicted target variable for the test set.
    :return: dataframe with the rank desired.
    """

    dtf_rank = pd.concat([X_test, pd.Series(y_test, name='TARGET_REAL'), pd.Series(y_pred, name='SCORE')], axis=1)
    dtf_rank.sort_values(by='SCORE', ascending=False, inplace=True)
    dtf_rank['RANK'] = list(range(1, dtf_rank.shape[0] + 1))

    # sorting
    lst_tmp = ['RANK', 'SCORE', 'TARGET_REAL']
    dtf_rank = dtf_rank[lst_tmp + [c for c in dtf_rank if c not in lst_tmp]].copy()

    # rounding for better understanding
    dtf_rank = dtf_rank.round(5)

    return dtf_rank
