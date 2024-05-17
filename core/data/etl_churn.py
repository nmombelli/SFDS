import logging
import numpy as np
import pandas as pd

from core.data.etl_utils import category_corr


def data_preparation_churn(dtf_load: pd.DataFrame) -> pd.DataFrame:
    """
        Data preparation for the CHURN dataset.
        :param dtf_load: the input dataset to prepare.
        :return: the parsed dataframe.
    """

    dtf_work = dtf_load.copy()

    # rename columns
    dtf_work.columns = [c.strip().upper() for c in dtf_work]
    dtf_work.rename({'ATTRITION_FLAG': 'TARGET'}, axis=1, inplace=True)

    # creating TARGET variable
    dtf_work['TARGET'] = np.where(dtf_work['TARGET'] == 'Attrited Customer', 1, 0)

    # set index
    dtf_work.set_index('CLIENTNUM', inplace=True)

    # columns to drop in the analysis:
    # GENDER: we want to avoid unfair scenarios
    # NAIVE_BAYES_CLASSIFIER ... : columns dropped as suggested by the author of the dataset.
    lst_drop = [
        'GENDER',
        'NAIVE_BAYES_CLASSIFIER_ATTRITION_FLAG_CARD_CATEGORY_CONTACTS_COUNT_12_MON_DEPENDENT_COUNT_EDUCATION_LEVEL_MONTHS_INACTIVE_12_MON_1',
        'NAIVE_BAYES_CLASSIFIER_ATTRITION_FLAG_CARD_CATEGORY_CONTACTS_COUNT_12_MON_DEPENDENT_COUNT_EDUCATION_LEVEL_MONTHS_INACTIVE_12_MON_2',
    ]
    dtf_work.drop(lst_drop, axis=1, inplace=True)

    # ENCODING
    dtf_work['EDUCATION_LEVEL'] = dtf_work['EDUCATION_LEVEL'].map({
        'Uneducated': 0,
        'Unknown': np.nan,
        'High School': 1,
        'College': 2,
        'Graduate': 3,
        'Post-Graduate': 4,
        'Doctorate': 5,
    })

    dtf_work['INCOME_CATEGORY'] = dtf_work['INCOME_CATEGORY'].map({
        'Unknown': np.nan,
        'Less than $40K': 1,
        '$40K - $60K': 2,
        '$60K - $80K': 3,
        '$80K - $120K': 4,
        '$120K +': 5,
    })

    # FORMATTING CATEGORICAL ENTRIES.
    # NAN values for categorical variables are labeled as UNKNOWN.
    dct_cat = {}
    lst_cat = dtf_work.select_dtypes(include='object').columns.tolist()

    # CORRELATION study
    dtf_cra = category_corr(dtf_in=dtf_work)
    dtf_corr = dtf_work[[c for c in dtf_work if c not in lst_cat]].corr()
    # dropping columns too correlated
    dtf_work.drop(['CREDIT_LIMIT'], axis=1, inplace=True)

    for c in lst_cat:
        dtf_work[c] = dtf_work[c].str.strip('.')
        dtf_work[c] = dtf_work[c].str.upper()
        dtf_work.loc[dtf_work[c] == 'UNKNOWN', c] = np.nan
        dct_cat[c] = set(dtf_work[c]) - {np.nan}

    # NAN evaluation
    dtf_work.dropna(axis=0, how='any', inplace=True)

    # Casting to dummy variables. First resulting column dropped after the encoding.
    dtf_work = pd.get_dummies(
        data=dtf_work,
        prefix=None,
        prefix_sep='_',
        dummy_na=False,
        columns=lst_cat,
        sparse=False,
        drop_first=True,
        dtype='int'
    )

    # printing the name of the columns dropped with the encoding
    for k, lst_val in dct_cat.items():
        for val in lst_val:
            if all(val not in c for c in dtf_work):
                logging.debug(f'{k} - DROPPED {val}')

    # sorting columns
    dtf_work = dtf_work[sorted(dtf_work.columns)].copy()

    return dtf_work
