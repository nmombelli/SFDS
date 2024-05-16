from core.data.etl_factory import etl_factory
from core.pipelines.ingestion import ingestion
from core.pipelines.splitting import split_data


def run_etl(str_source: str, split_strategy: str, n_splits: int, random_state: int):
    """
    Load the required dataset, apply data preparation and split the data in train and test sets.
    :param str_source: name of the source to load
    :param split_strategy: strategy to use when creating train and test sets. None or OVERSAMPLING allowed.
    :param n_splits: number of folds created for the cross validation.
    :param random_state: seed to be set for reproducibility
    :return: train and test sets with the indexes needed to perform cross validation
    """

    dtf_load = ingestion(str_source=str_source)

    # ETL and SPLITTING
    data_preparation = etl_factory(str_source)
    dtf_main = data_preparation(dtf_load=dtf_load)

    X_train, X_test, y_train, y_test, dct_cv = split_data(
        dtf_in=dtf_main,
        split_strategy=split_strategy,
        n_splits=n_splits,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, dct_cv
