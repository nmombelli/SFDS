import numpy as np
import pandas as pd

from safeaipackage.check_explainability import Explainability
from sklearn.ensemble import RandomForestClassifier


def xai_safe_global(X_train, X_test, y_train, y_test, model_params):

    model = RandomForestClassifier(**model_params)

    xe = Explainability(
        xtrain=X_train,
        xtest=X_test,
        ytrain=np.array(y_train),
        ytest=np.array(y_test),
        model=model,
    )

    xe.rge()

    return


def xai_safe_local(X_train, X_test, y_train, y_test, model_params, cust_id):

    X_test = pd.DataFrame(X_test.loc[cust_id].copy()).T
    y_test = pd.Series(y_test.loc[cust_id], index=[cust_id])

    model = RandomForestClassifier(**model_params)

    xe = Explainability(
        xtrain=X_train,
        xtest=X_test,
        ytrain=np.array(y_train),
        ytest=np.array(y_test),
        model=model,
    )

    xe.rge()

    return


if __name__ == '__main__':

    print('hello')


# TODO: warning shape
