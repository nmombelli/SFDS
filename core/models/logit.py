import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

# from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from core.pipelines.scaling import scaling_data


def stepwise_logit_frw(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dct_cv: dict,
        bln_scale: bool = True,
        pvalue: float = 0.05,
        maxiter: int = 5000,
        ths_delta_gain: float = 0.01,
        scoring: str = 'roc_auc',
) -> dict:
    """
    Stepwise Forward procedure for feature selection.
    Starting from a model with no features, at each iteration the feature providing the highest increment
    (compared to the previous iteration) with respect to the metric considered and having significant pvalue is added
    to the list of features to use in the model.
    The increment and the significance level are computed exploiting a cross validation process.
    The procedure stops when all features to add have not significant pvalue or do not provide increment more than
    ths_delta_gain.
    :param X_train: train dataset.
    :param y_train: target variable for training.
    :param dct_cv: dictionary with the indexes needed to create the folds during the cross validation phase.
    :param bln_scale: if True, data are scaled via standard scaling approach.
    :param pvalue: significance level of the test.
    :param maxiter: maximum number of iterations.
    :param ths_delta_gain: minimum level of increment needed to add the feature to the model.
    :param scoring: metric to evaluate the increment. Allowed values are roc_auc and f1_score.
    :return: dictionary with the statistics of the model.
    """

    if scoring not in ['f1', 'roc_auc']:
        raise ValueError('PLEASE SELECT A PROPER METRIC')

    lst_col_tst = X_train.columns.tolist()
    lst_col_sel = []
    dct_out = {'ALL_FEATURES': lst_col_tst}
    dct_metric = {'ITER_0': 0}
    bln_go = True
    counter = 1

    while bln_go:

        dct_out[counter] = {}

        lst_model_tmp = []
        logging.debug(f"ITER {counter} - SELECTED: {lst_col_sel} - TO TEST: {lst_col_tst}")

        for col in lst_col_tst:

            logging.debug(f"ITER {counter} - TESTING: {col}")
            lst_col_tmp = lst_col_sel + [col]
            lst_tmp_cv = []

            # CROSS VALIDATION
            for k, item in dct_cv.items():

                dct_tmp_cv = {'ITER': k}

                idx_train = item['TRAIN']
                idx_valid = item['VALIDATION']

                X_train_tmp = X_train.iloc[idx_train].copy()
                X_train_tmp = X_train_tmp[lst_col_tmp].copy()

                X_test_tmp = X_train.iloc[idx_valid].copy()
                X_test_tmp = X_test_tmp[lst_col_tmp].copy()

                y_train_tmp = y_train.iloc[idx_train].copy()
                y_test_tmp = y_train.iloc[idx_valid].copy()

                if bln_scale:
                    X_train_tmp, X_test_tmp = scaling_data(X_train=X_train_tmp, X_test=X_test_tmp)

                try:
                    lgt_class = sm.Logit(y_train_tmp, X_train_tmp)
                    model = lgt_class.fit(disp=False, maxiter=maxiter)
                    y_pred = model.predict(X_test_tmp)
                    dct_tmp_cv['PVALUE'] = model.pvalues
                    dct_tmp_cv['METRIC'] = roc_auc_score(y_test_tmp, y_pred)
                except Exception as e:
                    logging.error(f"CV ERROR - pvalue to nan and metric to 0.5 - {e}")
                    dct_tmp_cv['PVALUE'] = [np.nan] * len(X_train_tmp.columns)
                    dct_tmp_cv['METRIC'] = np.nan

                lst_tmp_cv.append(dct_tmp_cv)

            dtf_sum = pd.DataFrame.from_records({dct['ITER']: dct['PVALUE'] for dct in lst_tmp_cv})
            dtf_sum['MEDIAN'] = dtf_sum.apply(
                lambda x: x.median() if x.notna().sum() >= len(dct_cv) / 2 else np.nan, axis=1
            )
            dtf_sum.sort_values(by='MEDIAN', ascending=False, inplace=True)

            if all(dtf_sum['MEDIAN'] <= pvalue):
                metric = pd.Series([dct['METRIC'] for dct in lst_tmp_cv]).median()
            else:
                logging.debug(f"{col} - PVALUES NOT RELIABLE")
                metric = 0

            dct_tmp_cv = {
                'COLUMNS': lst_col_tmp,
                'TESTED': col,
                'METRIC_NAME': scoring,
                'METRIC_VALUE': metric,
            }

            lst_model_tmp.append(dct_tmp_cv)

        lst_model_tmp = sorted(lst_model_tmp, key=lambda d: d['METRIC_VALUE'], reverse=True)
        best_model = lst_model_tmp[0]
        dct_metric[f'ITER_{counter}'] = best_model['METRIC_VALUE']
        flt_gain = dct_metric[f'ITER_{counter}'] - dct_metric[f'ITER_{counter - 1}']

        best_col = best_model['TESTED']
        dct_out[counter]['GAIN'] = flt_gain

        if flt_gain <= ths_delta_gain:
            logging.debug(f'STOPPING the loop at ITER {counter} - not provided at least {ths_delta_gain} for {scoring}')
            dct_out[counter]['WINNER'] = None
            bln_go = False
        else:
            logging.debug(f"ITER: {counter} - Winner: {best_col} - NEW {scoring}: {dct_metric[f'ITER_{counter}']}")
            lst_col_sel.append(best_col)
            lst_col_tst.remove(best_col)
            dct_out[counter]['WINNER'] = best_col
            if not lst_col_tst:
                bln_go = False
            else:
                counter = counter + 1

    dct_out['COL_SELECT'] = lst_col_sel
    dct_out['COL_REMOVE'] = []

    logging.info(f"COLUMNS REMOVED: {[]}")
    logging.info(f"COLUMNS SELECTED: {lst_col_sel}")

    return dct_out


def stepwise_logit_back(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dct_cv: dict,
        bln_scale: bool = True,
        pvalue: float = 0.05,
        maxiter: int = 5000,
):
    """
    Stepwise backward procedure for feature selection.
    Starting from a model using all the features considered, at each iteration the feature with higher median pvalue is
    removed. The median is computed exploiting a cross validation process.
    The procedure stops when all pvalue are significant.
    :param X_train: train dataset.
    :param y_train: target variable for training.
    :param dct_cv: dictionary with the indexes needed to create the folds during the cross validation phase.
    :param bln_scale: if True, data are scaled via standard scaling approach.
    :param pvalue: significance level of the test.
    :param maxiter: maximum number of iterations.
    :return: dictionary with the statistics of the model.
    """

    lst_col_rem = []
    lst_col_sel = sorted(X_train.columns.tolist())

    bln_go = True
    counter = 1
    dct_out = {}

    logging.info(f'MODEL - STARTING FEATURES: {len(lst_col_sel)}')
    logging.debug(f'MODEL - STARTING FEATURES: {lst_col_sel}')

    while bln_go:

        dct_out[counter] = {}
        dct_out[counter]['SELECTED'] = lst_col_sel
        dct_out[counter]['REMOVED'] = lst_col_rem
        dct_tmp = {}

        for k, item in dct_cv.items():

            idx_train = item['TRAIN']
            idx_valid = item['VALIDATION']

            X_train_tmp = X_train.iloc[idx_train].copy()
            X_train_tmp = X_train_tmp[lst_col_sel].copy()
            y_train_tmp = y_train.iloc[idx_train].copy()

            X_test_tmp = X_train.iloc[idx_valid].copy()
            # y_test_tmp = y_train.iloc[idx_valid].copy()

            if bln_scale:
                X_train_tmp, X_test_tmp = scaling_data(X_train=X_train_tmp, X_test=X_test_tmp)

            try:
                lgt_class = sm.Logit(y_train_tmp, X_train_tmp)
                model = lgt_class.fit(disp=False, maxiter=maxiter)
                dct_tmp[k] = model.pvalues
            except Exception as e:
                logging.error(f"CV ERROR - pvalues to nan - {e}")
                dct_tmp[k] = [np.nan] * len(X_train_tmp.columns)

        dtf_sum = pd.DataFrame.from_dict(dct_tmp)
        dtf_sum['MEDIAN'] = dtf_sum.median(axis=1)
        dtf_sum.sort_values(by='MEDIAN', ascending=False, inplace=True)
        dct_out[counter] = dtf_sum

        if all(dtf_sum['MEDIAN'] <= pvalue):
            bln_go = False
        else:
            col_rem = dtf_sum['MEDIAN'].idxmax()
            pvl_max = dtf_sum['MEDIAN'].max()
            logging.info(f'DROPPED COLUMN {col_rem} - pvalue {pvl_max}')
            lst_col_sel.remove(col_rem)
            lst_col_rem.append(col_rem)

            if not lst_col_sel:
                raise KeyError('ALL FEATURES REMOVED')

            counter = counter + 1

    dct_out['COL_SELECT'] = lst_col_sel
    dct_out['COL_REMOVE'] = lst_col_rem
    
    logging.info(f"P-VALUES ARE GOOD")
    logging.info(f"COLUMNS REMOVED: {lst_col_rem}")
    logging.info(f"COLUMNS SELECTED: {lst_col_sel}")

    return dct_out
