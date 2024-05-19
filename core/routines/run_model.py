import numpy as np
import pandas as pd

from core.models.evaluation import evaluation
from core.models.forest import run_random_forest


def run_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        random_state: int = None
) -> tuple:
    """
    Run the model and predict the values for the test set.
    Optional: data can be scaled via standard scaling.
    Optional: feature selection via backward/forward stepwise procedure exploiting cross validation.
    :param X_train: train dataset
    :param X_test: test dataset
    :param y_train: target variable for training
    :param y_test:  target variable for testing.
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the predictions
    """

    if random_state:
        np.random.seed(random_state)

    # cross validation to fine tune
    model = run_random_forest(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state
    )

    # testing overfitting
    y_pred_train = model.predict(X_train)
    y_pred_proba_train = model.predict_proba(X_train)
    evaluation(y_train, y_pred_train, y_pred_proba_train, tpe='TRAIN', bln_save=True)

    # predicting on test
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    evaluation(y_test, y_pred, y_pred_proba, tpe='TEST', bln_save=True)

    test_set = pd.concat(
        [
            pd.Series(y_pred, index=X_test.index, name='MODEL_PRED'),
            pd.Series(round(y_pred_proba[:, 1], 4), index=X_test.index, name='MODEL_PRED_PROBA'),
            y_test,
            X_test,
        ],
        axis=1,
        verify_integrity=True,
    )

    return model, test_set
