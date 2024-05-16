import json
import logging
import os

from core.pipelines.output_preparation import join_output
from core.routines.run_etl import run_etl
from core.routines.run_model import run_model
from core.viz.viz_bar import viz_logit_barchart


def execute_main(
        str_source: str,
        split_strategy: str = None,
        n_splits: int = None,
        bln_scale: bool = True,
        logit_type: str = None,
        random_state: int = None,
) -> None:
    """
    Loading data, preparing data and running the logistic model
    :param str_source: name of the source to consider. Allowed values are HR and CHURN.
    :param split_strategy: how to split the dataset in train and test set. Allowed values are None and OVERSAMPLING.
    :param n_splits: number of folds used for cross validation.
    :param bln_scale: if True, train and test set are scaled with normal scaling approach.
    :param logit_type: how to make feature selection via logistic regression. Allowed values are:
        None: no particular strategy applied; the model uses all the features received.
        BACK: backward stepwise procedure applied to select only the relevant features.
        FORWARD: forward stepwise procedure applied to select only the relevant features.
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the test dataframe with its predictions.
    """

    X_train, X_test, y_train, y_test, dct_cv = run_etl(
        str_source=str_source,
        split_strategy=split_strategy,
        n_splits=n_splits,
        random_state=random_state
    )

    # Creating a copy of the X sets before scaling (in the model below). We want to preserve the real values for plots.
    X_test_real = X_test.copy()

    dct_out = run_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dct_cv=dct_cv,
        bln_scale=bln_scale,
        logit_type=logit_type,
        random_state=random_state,
    )

    # Preparing data to help the user to visualize the ranking.
    # Real values (not scaled) are kept.
    dtf_rank = join_output(
        X_test=X_test_real[dct_out['LST_COL_SEL']],
        y_test=y_test,
        y_pred=dct_out['Y_PRED'],
    )

    # Preparing data for the plotting phase.
    # In case of bln_scale=True, scaled values are used. Otherwise, the output is the same as dtf_rank
    dtf_exp = join_output(
        X_test=dct_out['X_TEST'][dct_out['LST_COL_SEL']],
        y_test=y_test,
        y_pred=dct_out['Y_PRED'],
    )
    dtf_exp.drop('TARGET_REAL', axis=1, inplace=True)

    viz_logit_barchart(dct_coef=dct_out['DCT_COEF'], figsize=(30, 12), maxnum_feat=30)

    # storing ranking data for user's explorations.
    str_path_viz = os.environ['PATH_OUT_VIZ']
    str_path_mod = os.environ['PATH_OUT_MOD']
    dtf_rank.to_csv(f"{str_path_viz}RANKING_{str_source}.csv", sep=';', index=True)
    logging.info(f"RANKING table dumped in {str_path_viz}")
    with open(f"{str_path_mod}COEF_LOGIT_{str_source}.json", 'w') as f:
        json.dump(dct_out['DCT_COEF'], f)
    logging.info(f"COEF_LOGIT table dumped in {str_path_mod}")

    return
