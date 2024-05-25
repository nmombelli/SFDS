import pandas as pd
import os
import json

from core.routines.run_etl import run_etl
from core.routines.run_model import run_model


def execute_main(
        str_source: str,
        split_strategy: str = None,
        bln_scale: bool = False,
        random_state: int = None,
) -> dict:
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

    model, test_set = run_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )

    model_params = model.get_params()

    with open(os.environ['PATH_OUT_MOD'] + 'model_params.json', 'w') as fp:
        json.dump(model_params, fp, indent=4)

    pd.to_pickle(model, os.environ['PATH_OUT_MOD'] + 'model.pickle')
    pd.to_pickle(X_train, os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
    pd.to_pickle(X_test, os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    pd.to_pickle(y_train, os.environ['PATH_OUT_MOD'] + 'y_train.pickle')
    pd.to_pickle(y_test, os.environ['PATH_OUT_MOD'] + 'y_test.pickle')
    test_set.to_csv(os.environ['PATH_OUT_MOD'] + 'test_set.csv', sep=';', index=True)

    return model_params


if __name__ == '__main__':

    import logging

    from setting.environment import set_env
    from setting.logger import set_logger

    set_env()
    set_logger(level='INFO')

    execute_main(str_source='CHURN', split_strategy='OVERSAMPLING')
    logging.info('I AM DONE')
