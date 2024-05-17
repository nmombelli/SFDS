import logging

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay


def evaluation(y_true, y_pred, y_pred_proba, tpe):

    auc_test = round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
    accuracy_test = round(accuracy_score(y_true, y_pred), 4)
    precision_test = round(precision_score(y_true, y_pred), 4)
    recall_test = round(recall_score(y_true, y_pred), 4)
    f1_score_test = round(f1_score(y_true, y_pred), 4)

    logging.info(f'AUC: {tpe} set: {auc_test}')
    logging.info(f'ACCURACY: {tpe} set: {accuracy_test}')
    logging.info(f'PRECISION: {tpe} set: {precision_test}')
    logging.info(f'RECALL: {tpe} set: {recall_test}')
    logging.info(f'F1 SCORE: {tpe} set: {f1_score_test}')

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    return
