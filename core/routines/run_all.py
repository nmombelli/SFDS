import pandas as pd
import os

from core.routines.run_etl import run_etl
from core.routines.run_model import run_model


def execute_main(
        str_source: str,
        split_strategy: str = None,
        bln_scale: bool = True,
        random_state: int = None,
) -> None:
    """
    Loading data, preparing data and running the classification model
    :param str_source: name of the source to consider. Allowed values are HR and CHURN.
    :param split_strategy: how to split the dataset in train and test set. Allowed values are None and OVERSAMPLING.
    :param bln_scale: if True, train and test set are scaled with normal scaling approach.
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the test dataframe with its predictions.
    """

    X_train, X_test, y_train, y_test = run_etl(
        str_source=str_source,
        split_strategy=split_strategy,
        bln_scale=bln_scale,
        random_state=random_state
    )

    # Creating a copy of the X sets before scaling (in the model below). We want to preserve the real values for plots.
    # X_test_real = X_test.copy()

    model, y_pred = run_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )

    pd.to_pickle(model, os.environ['PATH_OUT_MOD'] + 'model.pickle')
    pd.to_pickle(X_train, os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
    pd.to_pickle(X_test, os.environ['PATH_OUT_MOD'] + 'X_test.pickle')

    return


if __name__ == '__main__':

    import logging

    execute_main(str_source='CHURN')
    logging.info('I AM DONE')

    # TODO: please fix CV numbers of iteration, fix the number of rows in shap dtf
