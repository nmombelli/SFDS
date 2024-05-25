import pandas as pd

from safeaipackage.check_explainability import Explainability
from sklearn.ensemble import RandomForestClassifier


def xai_safe_global(X_train, X_test, y_train, y_test, model_params):
    """
    Explain global dataset with SAFE approach:
    :param X_train: dataset needed for train "model";
    :param X_test: dataset for explanations;
    :param y_train: target variable needed to train model;
    :param y_test: target variable for explanations;
    :param model_params: the model to train and explain;
    :return: None
    """
    model = RandomForestClassifier(**model_params)

    xe = Explainability(
        xtrain=X_train,
        xtest=X_test,
        ytrain=y_train,
        ytest=y_test,
        model=model,
    )

    # The warning raised in here is due to what happens in the function.
    # They use a reset_index(drop=True) on the train part. Bad move. You cannot fix this outside the package
    xe.rge()

    return


def xai_safe_local(X_train, X_test, y_train, y_test, model_params, cust_id):
    """
    Explain local instance of a customer with SAFE approach:
    :param X_train: dataset needed for train "model";
    :param X_test: dataset for explanations. It is filtered with cust_id;
    :param y_train: target variable needed to train model;
    :param y_test: target variable for explanations. It is filtered with cust_id;
    :param model_params: the model to train and explai;
    :param cust_id: the customer ID to explain;
    :return: None
    """

    X_test = pd.DataFrame(X_test.loc[cust_id].copy()).T
    y_test = pd.Series(y_test.loc[cust_id], index=[cust_id])

    model = RandomForestClassifier(**model_params)

    xe = Explainability(
        xtrain=X_train,
        xtest=X_test,
        ytrain=y_train,
        ytest=y_test,
        model=model,
    )

    # The warning raised in here is due to what happens in the function.
    # They use a reset_index(drop=True) on the train part. Bad move. You cannot fix this outside the package
    xe.rge()

    return


if __name__ == '__main__':

    print('hello')
