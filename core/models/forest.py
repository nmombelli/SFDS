import logging
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def run_random_forest(X_train, y_train, random_state):

    logging.info(f'TRAINING_PHASE STARTED')

    with open('config/param_forest.yaml') as f:
        dct_param_rf = yaml.safe_load(f)['RANDOM_FOREST']

    with open('config/param_cv.yaml') as f:
        dct_param_cv = yaml.safe_load(f)

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=random_state)

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=dct_param_rf,
        random_state=random_state,
        refit=True,
        **dct_param_cv
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    dct_cv_results = rand_search.cv_results_
    logging.debug(f'CV - mean_train_score: {dct_cv_results["mean_train_score"]}')
    logging.debug(f'CV - mean_test_score: {dct_cv_results["mean_test_score"]}')

    # Print the best hyperparameters
    logging.info(f'Best hyperparameters: {rand_search.best_params_}')

    # Create a variable for the best model.
    # Attention point:
    # thanks to parameter refit=True in RandomizedSearchCV, the best_estimator_ is already fit on the full training set.
    model = rand_search.best_estimator_

    logging.info(f'TRAINING_PHASE COMPLETED')

    return model
