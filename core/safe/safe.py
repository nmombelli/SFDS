import pandas as pd
import os

from matplotlib import pyplot as plt
from safeaipackage.check_accuracy import Accuracy
from safeaipackage.check_explainability import Explainability
from safeaipackage.check_fairness import Fairness
from safeaipackage.check_robustness import Robustness

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

scores = {}

# ACCURACY
safe_acc = Accuracy(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
scores['RGA'] = round(safe_acc.rga(), 4)

# ROBUSTNESS
safe_rob = Robustness(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
scores['RGR'] = round(safe_rob.rgr_all(perturbation_percentage=0.05), 4)

# FAIRNESS
safe_fair = Fairness(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
rgf = safe_fair.rgf(protectedvariable=['GENDER_M'])
scores['RGF'] = round(rgf.loc['GENDER_M', 'RGF'], 4)

# EXPLAINABILITY
safe_xai = Explainability(model=model, xtrain=X_train, xtest=X_test, ytrain=y_train, ytest=y_test)
rge_df = safe_xai.rge()
scores['RGE'] = list(rge_df.to_dict().values())[0]
scores['RGE'] = {k: round(v, 4) for k, v in scores['RGE'].items()}

plt.figure()
plt.barh(rge_df.index, rge_df["RGE"], color='orange')
plt.xlabel("RGE (Feature Importance)")
plt.ylabel("Feature")
# plt.title("RGE")
plt.tight_layout()
plt.savefig(f'{os.environ['PATH_OUT_SAFE']}/GLOBAL.png', )
plt.close()


# STATISTICAL TESTS

safe_acc.rga_statistic_test(problemtype='classification')
safe_rob.rgr_statistic_test(problemtype='classification')
safe_fair.rgf_statistic_test(protectedvariable=['GENDER_M'])

dct_exp = {}
for column in X_train.columns.tolist():
    dct_exp[column] = safe_xai.rge_statistic_test(variable=column)
