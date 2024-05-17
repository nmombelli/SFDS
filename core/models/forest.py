import logging
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score


def run_random_forest(X_train, y_train, n_splits, random_state):

    print(y_train.value_counts(normalize=True))

    with open('config/param_forest.yaml') as f:
        dct_param = yaml.safe_load(f)['RANDOM_FOREST']

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=random_state)

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=dct_param,
        n_iter=5,
        cv=n_splits,
        random_state=random_state,
        scoring='roc_auc',
        return_train_score=True,
        verbose=0
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    dct_cv_results = rand_search.cv_results_
    logging.debug(f'CV - mean_train_score: {dct_cv_results["mean_train_score"]}')
    logging.debug(f'CV - mean_test_score: {dct_cv_results["mean_test_score"]}')

    # Create a variable for the best model
    model = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

    return model
