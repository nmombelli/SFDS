import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import roc_auc_score

from core.models.logit import stepwise_logit_back
from core.models.logit import stepwise_logit_frw
from core.pipelines.scaling import scaling_data


def run_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        dct_cv: dict,
        bln_scale: bool,
        logit_type: str = None,
        random_state: int = None
) -> dict:
    """
    Run the logistic regression model and predict the values for the test set.
    Optional: data can be scaled via standard scaling.
    Optional: feature selection via backward/forward stepwise procedure exploiting cross validation.
    :param X_train: train dataset
    :param X_test: test dataset
    :param y_train: target variable for training
    :param y_test:  target variable for testing
    :param dct_cv: dictionary with the indexes needed to create the folds during the cross validation phase
    :param bln_scale: if True, data are scaled via standard scaling approach
    :param logit_type: how to make feature selection via logistic regression. Allowed values are:
        None: no particular strategy applied; the model uses all the features received.
        BACK: backward stepwise procedure applied to select only the relevant features.
        FORWARD: forward stepwise procedure applied to select only the relevant features.
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the predictions
    """

    logging.debug(f'MODEL: bln_scale={bln_scale}')

    if random_state:
        np.random.seed(random_state)

    # running stepwise logistic regression to select the most relevant features.
    if logit_type == 'FORWARD':
        dct_lgt = stepwise_logit_frw(
            X_train=X_train,
            y_train=y_train,
            dct_cv=dct_cv,
            bln_scale=bln_scale,
            pvalue=0.05,
            maxiter=5000,
            ths_delta_gain=0.001,
            scoring='roc_auc',
        )
    elif logit_type == 'BACK':
        dct_lgt = stepwise_logit_back(
            X_train=X_train,
            y_train=y_train,
            dct_cv=dct_cv,
            bln_scale=bln_scale,
            pvalue=0.05,
            maxiter=2000
        )
    elif logit_type is None:
        dct_lgt = {'COL_SELECT': X_train.columns.tolist()}
    else:
        raise ValueError(f"logit_type {logit_type} NOT SUPPORTED")

    # Now run the logistic model using only the feature selected above.
    # Data are to be scaled (if requested) to stay consistent with the stepwise procedure.
    if bln_scale:
        X_train, X_test = scaling_data(X_train=X_train, X_test=X_test)

    lst_col = dct_lgt['COL_SELECT']
    lgt_class = sm.Logit(y_train, X_train[lst_col])
    model = lgt_class.fit(disp=False, maxiter=5000)
    dct_coef = {k: round(v, 2) for k, v in sorted(model.params.items())}

    # testing overfitting
    y_pred_train = model.predict(X_train[lst_col])
    metric_train = round(roc_auc_score(y_train, y_pred_train), 4)
    logging.info(f'AUC train set: {metric_train}')

    # predicting on test
    y_pred = model.predict(X_test[lst_col])
    metric_test = round(roc_auc_score(y_test, y_pred), 4)
    logging.info(f'AUC test set: {metric_test}')

    dct_out = {
        'DCT_COEF': dct_coef,
        'X_TEST': X_test,
        'Y_PRED': y_pred,
        'LST_COL_SEL': lst_col
    }

    return dct_out
