import shap


def xai_global_shap(shap_values, X_test):

    feature_names = X_test.columns.tolist()

    # Plot the summary plot for global explanation
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    # Optional: Plot individual feature importance
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

    # Optional: Dependency plot for a specific feature
    # shap.dependence_plot('AVG_OPEN_TO_BUY', shap_values, X_test)

    return


def xai_local_shap(shap_values, idx):

    shap.plots.bar(shap_values[idx])
    shap.plots.waterfall(shap_values[idx])

    return
