import lime
import lime.lime_tabular
import pandas as pd


model = pd.read_pickle('C:/Users/NMOMBELLI/Desktop/SFDS/CHURN/MODEL/model.pickle')
X_train = pd.read_pickle('C:/Users/NMOMBELLI/Desktop/SFDS/CHURN/MODEL/X_train.pickle')
X_test = pd.read_pickle('C:/Users/NMOMBELLI/Desktop/SFDS/CHURN/MODEL/X_test.pickle')

# LIME has one explainer for all the models
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.values.tolist(),
    class_names=['PINO'],
    verbose=True,
    mode='classification'
)

j = 5
exp = explainer.explain_instance(X_test.values[j], model.predict, num_features=6)
