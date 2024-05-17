import os
import shap

from core.routines.run_etl import run_etl
from core.routines.run_model import run_model
from core.xai.xai_shap import xai_global_shap, xai_local_shap


def execute_main(
        str_source: str,
        split_strategy: str = None,
        n_splits: int = None,
        bln_scale: bool = True,
        random_state: int = None,
) -> tuple:
    """
    Loading data, preparing data and running the logistic model
    :param str_source: name of the source to consider. Allowed values are HR and CHURN.
    :param split_strategy: how to split the dataset in train and test set. Allowed values are None and OVERSAMPLING.
    :param n_splits: number of folds used for cross validation.
    :param bln_scale: if True, train and test set are scaled with normal scaling approach.
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
    # X_test_real = X_test.copy()

    model, y_pred = run_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_splits=n_splits,
        bln_scale=bln_scale,
        random_state=random_state,
    )

    return model, X_test


if __name__ == '__main__':
    import yaml
    from setting.logger import set_logger

    print('I AM READY')

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
        n_splits=5,
        split_strategy='OVERSAMPLING',
    )
    print('I AM DONE')

    # Use the SHAP library to explain the model's predictions
    explainer = shap.Explainer(model_out.predict, X_test_out)
    shap_values = explainer(X_test_out)

    xai_global_shap(shap_values=shap_values, X_test=X_test_out)
    xai_local_shap(shap_values=shap_values, idx=None)

    print('OVER')
