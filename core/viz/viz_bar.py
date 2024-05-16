import logging
import os

from matplotlib import pyplot as plt


def viz_logit_barchart(dct_coef: dict, figsize: tuple = None, maxnum_feat: int = 10) -> None:

    """
    Plotting the barchart of the coefficient of the logistic regression model.
    :param dct_coef: dictionary with the coefficients of the model
    :param figsize: size of the figure to plot.
    :param maxnum_feat: maximum number of features to plot. This prevents from having too complicated plots.
    :return:
    """

    str_path_out = os.environ['PATH_OUT_MOD']

    if len(dct_coef) > maxnum_feat:
        dct_coef = dict(sorted(dct_coef.items(), key=lambda item: abs(item[1]), reverse=True)[:maxnum_feat])

    colors = []
    for k in dct_coef:
        if k == 'INTERCEPT':
            colors.append('#ed872d')  # orange
        elif dct_coef[k] >= 0:
            colors.append('#73e2f7')  # tiffany
        else:
            colors.append('#e1c4ff')  # lilac

    ymax = max(dct_coef.values())
    ymin = min(dct_coef.values())

    if not figsize:
        figsize = (3 * len(dct_coef), 15)

    fig, ax = plt.subplots(figsize=figsize)
    bar_container = ax.bar(dct_coef.keys(), dct_coef.values())

    yoff = 0.2
    plt.ylim(min(ymin + yoff * ymin, 0), max(ymax + yoff * ymax, 0))
    plt.yticks(fontsize=20)

    y_pos_offset = 0.03

    for bar, color, value in zip(bar_container, colors, dct_coef.values()):
        bar.set_color(color)
        bar.set_edgecolor('grey')
        bar.set_linewidth(0.5)
        if value > 0:
            y_pos = bar.get_height() + y_pos_offset
        else:
            y_pos = bar.get_height() - y_pos_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            round(value, 2),
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=15
        )

    plt.xticks(rotation=30, ha='right', fontsize=15)
    # plt.title('Logistic Regression Model - Coefficients', y=1.05, fontweight='bold')
    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    plt.savefig(str_path_out + f'BARCHART_LOGIT.png', bbox_inches='tight')

    logging.info(f"BARCHART table dumped in {str_path_out}")

    return
