import logging
import numpy as np
import pandas as pd

from core.models.evaluation import evaluation
from core.models.forest import run_random_forest
# from core.pipelines.scaling import scaling_data


def run_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_splits: int,
        bln_scale: bool,
        random_state: int = None
) -> tuple:
    """
    Run the model and predict the values for the test set.
    Optional: data can be scaled via standard scaling.
    Optional: feature selection via backward/forward stepwise procedure exploiting cross validation.
    :param X_train: train dataset
    :param X_test: test dataset
    :param y_train: target variable for training
    :param y_test:  target variable for testing
    :param n_splits: number of folds used for cross validation.
    :param bln_scale: if True, data are scaled via standard scaling approach
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the predictions
    """

    logging.debug(f'MODEL: bln_scale={bln_scale}')

    if random_state:
        np.random.seed(random_state)

    # cross validation to fine tune
    model = run_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_splits=n_splits,
        random_state=random_state
    )

    # testing overfitting
    y_pred_train = model.predict(X_train)
    y_pred_proba_train = model.predict_proba(X_train)
    evaluation(y_train, y_pred_train, y_pred_proba_train, tpe='TRAIN')

    # predicting on test
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    evaluation(y_test, y_pred, y_pred_proba, tpe='TEST')

    return model, y_pred
