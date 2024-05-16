import logging
import numpy as np
import pandas as pd

# from core.data.etl_utils import category_corr


def data_preparation_hr(dtf_load: pd.DataFrame) -> pd.DataFrame:
    """
    Data preparation for the HR dataset.
    :param dtf_load: the input dataset to prepare.
    :return: the parsed dataframe.
    """

    dtf_work = dtf_load.copy()

    # rename columns
    dtf_work.columns = dtf_work.columns.str.upper()
    dtf_work.rename({'SL_NO': 'ID', 'SPECIALISATION': 'SPEC'}, axis=1, inplace=True)

    # creating TARGET variable
    dtf_work['TARGET'] = np.where(dtf_work['STATUS'] == 'Placed', 1, 0)

    # setting index
    dtf_work['ID'] = dtf_work['ID'].apply(lambda x: str(x).zfill(5))
    dtf_work.set_index(['ID'], inplace=True)

    # columns to drop in the analysis:
    # STATUS and SALARY: old target, now replaced
    # GENDER: we want to avoid unfair scenarios
    lst_drop = [
        'STATUS',
        'SALARY',
        'GENDER',
    ]
    dtf_work.drop(lst_drop, axis=1, inplace=True)

    # NAN evaluation
    logging.debug(f"NAN FOUND: {dtf_work.isna().sum().to_dict()}")

    # study of the correlation between categorical features
    # dtf_cra = category_corr(dtf_in=dtf_work)

    # CLEANING CATEGORICAL FEATURES
    lst_col = dtf_work.columns.tolist()
    lst_cat = dtf_work.select_dtypes(include='object').columns.tolist()
    dct_cat = {}

    for c in lst_cat:
        logging.debug(f'Fixing values style for {c}')
        dtf_work[c] = dtf_work[c].str.upper()
        dct_cat[c] = set(dtf_work[c])

    dtf_work['HSC_S'] = dtf_work['HSC_S'].str[:3]

    # Casting to dummy variables. No columns dropped after the encoding yet.
    dtf_work = pd.get_dummies(
        data=dtf_work,
        prefix=None,
        prefix_sep='_',
        dummy_na=False,
        columns=lst_cat,
        sparse=False,
        drop_first=False,
        dtype='int'
    )

    # Once a feature is turned to binary, the resulting column less correlated with the target is dropped
    dtf_corr = dtf_work.corr()
    for k in dct_cat:
        ser_corr = dtf_corr.loc[[c for c in dtf_corr if k in c], 'TARGET']
        col_drop = abs(ser_corr).idxmin()
        logging.debug(f'DROP {col_drop} - LESS CORRELATED WITH TGT')
        dtf_work.drop(col_drop, axis=1, inplace=True)

    logging.debug(f'Created dummy columns: {[c for c in dtf_work if c not in lst_col]}')

    # sorting columns
    dtf_work = dtf_work[sorted(dtf_work.columns)].copy()

    return dtf_work
