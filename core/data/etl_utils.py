import numpy as np
import pandas as pd
import scipy.stats as ss


def cramers_corrected_stat(confusion_matrix):
    """
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def category_corr(dtf_in: pd.DataFrame):
    """
    Compute the correlation between categorical variables using the cramers_corrected_stat method.
    :param dtf_in: dataframe to study
    :return: dataframe of the correlations
    """

    lst_cat = dtf_in.select_dtypes(include='object').columns.tolist()

    if 'TARGET' in dtf_in:
        lst_cat = lst_cat + ['TARGET']

    lst_cra = []
    for c1 in lst_cat:
        for c2 in lst_cat:
            dct_cra = {'COLUMN_1': c1, 'COLUMN_2': c2}
            confusion_matrix = pd.crosstab(dtf_in[c1], dtf_in[c2])
            dct_cra['CRAMER'] = cramers_corrected_stat(confusion_matrix)
            lst_cra.append(dct_cra)

    dtf_cra = pd.DataFrame.from_records(lst_cra)
    dtf_cra = pd.pivot_table(dtf_cra, index='COLUMN_1', values='CRAMER', columns='COLUMN_2')

    return dtf_cra
