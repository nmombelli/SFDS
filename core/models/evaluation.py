import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay


def evaluation(y_true: np.array, y_pred: np.array, y_pred_proba: np.array, tpe: str, bln_save: bool):

    auc_test = round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
    accuracy_test = round(accuracy_score(y_true, y_pred), 4)
    precision_test = round(precision_score(y_true, y_pred), 4)
    recall_test = round(recall_score(y_true, y_pred), 4)
    f1_score_test = round(f1_score(y_true, y_pred), 4)

    logging.info(f'{tpe} set: AUC: {auc_test}')
    logging.info(f'{tpe} set: ACCURACY: {accuracy_test}')
    logging.info(f'{tpe} set: PRECISION: {precision_test}')
    logging.info(f'{tpe} set: RECALL: {recall_test}')
    logging.info(f'{tpe} set: F1 SCORE: {f1_score_test}')

    if bln_save:
        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.savefig(f'{os.environ['PATH_OUT_VIZ']}/CONFUSION_MATRIX_{tpe}.png')
        plt.close()

    return
