from core.data.etl_churn import data_preparation_churn
from core.data.etl_hr import data_preparation_hr


def etl_factory(str_source):
    """
    Retrieve the data preparation method related to the desired source
    :param str_source: name of the source loaded
    :return: needed method
    """

    if str_source == 'HR':
        data_preparation = data_preparation_hr
    elif str_source == 'CHURN':
        data_preparation = data_preparation_churn
    else:
        raise ValueError('CANNOT MAKE DATA PREPARATION FOR', str_source)

    return data_preparation
