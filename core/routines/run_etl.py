import logging

from core.data.etl_factory import etl_factory
from core.pipelines.ingestion import ingestion
from core.pipelines.scaling import scaling_data
from core.pipelines.splitting import split_data


def run_etl(str_source: str, split_strategy: str, bln_scale: bool, random_state: int):
    """
    Load the required dataset, apply data preparation and split the data in train and test sets.
    :param str_source: name of the source to load
    :param split_strategy: strategy to use when creating train and test sets. None or OVERSAMPLING allowed.
    :param bln_scale: if True, data are scaled via standard scaling approach
    :param random_state: seed to be set for reproducibility.
    :return: train and test sets with the indexes needed to perform cross validation
    """

    dtf_load = ingestion(str_source=str_source)

    # ETL and SPLITTING
    data_preparation = etl_factory(str_source)
    dtf_main = data_preparation(dtf_load=dtf_load)

    X_train, X_test, y_train, y_test = split_data(
        dtf_in=dtf_main,
        split_strategy=split_strategy,
        random_state=random_state,
    )

    if bln_scale:
        logging.debug(f'MODEL: bln_scale={bln_scale}')
        X_train, X_test = scaling_data(X_train=X_train, X_test=X_test)

    return X_train, X_test, y_train, y_test
