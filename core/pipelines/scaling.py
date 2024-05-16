import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler


def scaling_data(X_train, X_test):
    """
    Scaling train and test set by using the standard scaling approach. Boolean variables are not scaled.
    :param X_train: train set to fit (and transform) the scaler.
    :param X_test: test set to transform.
    :return: train and test set scaled.
    """

    lst_num = X_train.select_dtypes(include='number').columns.tolist()
    if set(lst_num) != set(X_train):
        raise KeyError('COLUMNS ARE NOT ALL NUMERIC')

    lst_bin = []
    for c in lst_num:
        if set(X_train[c]).issubset({0, 1}):
            lst_bin.append(c)
    lst_flt = [c for c in lst_num if c not in lst_bin]

    if set(lst_bin + lst_flt) != set(X_train):
        raise KeyError('WRONG Columns split')

    # Scaling only if there are features to scale. Condition necessary to avoid error
    if lst_flt:

        logging.debug(f'Scaling features: {lst_flt}')

        # fitting
        sc = StandardScaler()
        sc.fit(X_train[lst_flt])

        # transforming
        X_train = pd.concat(
            [pd.DataFrame(sc.transform(X_train[lst_flt]), columns=lst_flt, index=X_train.index), X_train[lst_bin]],
            axis=1,
            verify_integrity=True
        )
        X_test = pd.concat(
            [pd.DataFrame(sc.transform(X_test[lst_flt]), columns=lst_flt, index=X_test.index), X_test[lst_bin]],
            axis=1,
            verify_integrity=True
        )

    return X_train, X_test
