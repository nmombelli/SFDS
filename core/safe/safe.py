import pandas as pd
import os

from safeaipackage.check_accuracy import Accuracy
from safeaipackage.check_robustness import Robustness
from safeaipackage.check_fairness import Fairness

from setting.environment import set_env
from setting.logger import set_logger


# TODO: all this must become a (or more) method call by API

set_env()
set_logger(level='INFO')

model = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'model.pickle')
X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
y_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_train.pickle')
y_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_test.pickle')


# ACCURACY
safe_acc = Accuracy(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
rga = safe_acc.rga()
# safe_acc.rga_statistic_test(problemtype='classification')

# ROBUSTNESS
safe_rob = Robustness(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
rgr = safe_rob.rgr_all(perturbation_percentage=0.05)
# safe_rob.rga_statistic_test(problemtype='classification')

# FAIRNESS
safe_fair = Fairness(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
rgf = safe_fair.rgf(protectedvariable=['GENDER_M'])
rgf_gender = rgf.loc['GENDER_M', 'RGF']
