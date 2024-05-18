import numpy as np
import os
import shap

from core.routines.run_etl import run_etl
from core.routines.run_model import run_model
from core.xai.xai_shap import xai_global_shap, xai_local_shap


def execute_main(
        str_source: str,
        split_strategy: str = None,
        bln_scale: bool = True,
        random_state: int = None,
) -> tuple:
    """
    Loading data, preparing data and running the logistic model
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

    return model, X_test


if __name__ == '__main__':

    import logging
    import yaml
    from setting.logger import set_logger

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']
    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    os.makedirs(str_path_viz, exist_ok=True)
    os.makedirs(str_path_mod, exist_ok=True)

    model_out, X_test_out = execute_main(
        str_source='CHURN',
        split_strategy='OVERSAMPLING',
        bln_scale=False,
    )

    X_test_out = X_test_out.iloc[:10]

    os.makedirs(f'{os.environ['PATH_OUT_VIZ']}/SHAP', exist_ok=True)

    # Use the SHAP library to explain the model's predictions
    explainer = shap.Explainer(model_out.predict, X_test_out)
    shap_values = explainer(X_test_out)

    xai_global_shap(shap_values=shap_values, X_test=X_test_out, bln_save=True)

    i = np.random.randint(low=0, high=shap_values.shape[0])
    xai_local_shap(shap_value_cust=shap_values[i], cust_id=X_test_out.index.tolist()[i], bln_save=True)
    i = np.random.randint(low=0, high=shap_values.shape[0])
    xai_local_shap(shap_value_cust=shap_values[i], cust_id=X_test_out.index.tolist()[i], bln_save=True)

    logging.info('I AM DONE')
