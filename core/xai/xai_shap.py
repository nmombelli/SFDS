import matplotlib.pyplot as plt
import os
import pandas as pd
import shap


def xai_global_shap(shap_values, X_test: pd.DataFrame, bln_save: True):
    """
    Create SHAP global plots and save them
    :param shap_values: shapley values needed for the plot
    :param X_test: dataset on which compute the global contributions.
    :param bln_save: whether to store the plot or not
    :return:
    """

    feature_names = X_test.columns.tolist()

    # Plot the summary plot for global explanation
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_size=(15, 10))

    if bln_save:
        plt.savefig(f'{os.environ['PATH_OUT_VIZ']}/GLOBAL_BEESWARM.png')
        plt.close()

    # Optional: Plot individual feature importance
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", plot_size=(15, 10))

    if bln_save:
        plt.savefig(f'{os.environ['PATH_OUT_VIZ']}/GLOBAL_BARCHART.png')
        plt.close()

    # Optional: Dependency plot for a specific feature
    # shap.dependence_plot('AVG_OPEN_TO_BUY', shap_values, X_test)

    return


def xai_local_shap(shap_values, idx: int, bln_save: bool):
    """
    Create SHAP local plots and save them.
    :param shap_values: shapley values needed for the plot
    :param idx: index of the instance to plot.
    :param bln_save: whether to store the plot or not
    :return:
    """

    shap.plots.bar(shap_values[idx])

    if bln_save:
        plt.savefig(f'{os.environ['PATH_OUT_VIZ']}/{idx}_BARCHART.png')
        plt.close()

    shap.plots.waterfall(shap_values[idx])

    if bln_save:
        plt.savefig(f'{os.environ['PATH_OUT_VIZ']}/{idx}_WATERFALL.png')
        plt.close()

    return
