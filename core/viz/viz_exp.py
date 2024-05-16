import logging
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt


dct_level_detail = {
    'ALL': 1,
    'HIGH': 0.9,
    'MEDIUM': 0.7,
    'LOW': 0.5
}


def explainer_delta_rank(
        ser_h: pd.Series,
        ser_l: pd.Series,
        dct_coef: dict,
        scale: str = None,
        level_detail: str = 'ALL',
        maxnum_feat: int = 10,
):

    """
    Evaluative item contrastive graph between two items.
    :param ser_h: features of the item with higher rank.
    :param ser_l: feature of the item with lower rank.
    :param dct_coef: dictionary with the coefficient of the logistic model.
    :param scale: the scaling method applied to the features before plotting. Allowed values are None, log and prc (%).
    :param level_detail: the amount of information explaining the ranking difference that the user wants to visualize.
        Allowed values are LOW, MEDIUM, HIGH, ALL, each corresponding to a certain percentage.
        Each feature provides a certain contribution in defining the overall item difference in ranking.
        This percentage represent how much of the overall contribution I want to explode (and hence consider the
        features whose summed contribution reaches the percentage).
    :param maxnum_feat: maximum number of features to plot. This prevents from having too complicated plots.
    :return:
    """

    str_path_out = os.environ['PATH_OUT_VIZ']

    if level_detail not in ['ALL', 'HIGH', 'MEDIUM', 'LOW']:
        raise ValueError("level_detail not in ['HIGH', 'MEDIUM', 'LOW']")

    if scale not in ['log', 'prc', None]:
        raise ValueError("scale not in ['log', 'prc', None]")

    level_ths = dct_level_detail[level_detail]

    if not (isinstance(ser_h, pd.Series) and isinstance(ser_l, pd.Series)):
        raise TypeError('ser_h and ser_l are not pd.Series')

    rank_h = int(ser_h['RANK'])
    rank_l = int(ser_l['RANK'])

    if rank_h >= rank_l:
        raise ValueError('Ranks NOT in correct order')

    ser_h.names = [ser_h.name, ser_h['RANK'], ser_h['SCORE']]
    ser_l.names = [ser_l.name, ser_l['RANK'], ser_l['SCORE']]
    ser_h = ser_h.drop(['RANK', 'SCORE', 'TARGET_REAL'])
    ser_l = ser_l.drop(['RANK', 'SCORE', 'TARGET_REAL'])

    if set(ser_h.index) != set(ser_l.index):
        raise ValueError('ser_h and ser_l have different indexes')

    if set(ser_h.index) != set(dct_coef.keys()):
        raise ValueError('series and dictionary have different indexes')

    ser_dlt = pd.Series(ser_h - ser_l, name='DELTA')
    ser_coe = pd.Series(dct_coef, name='COEFFICIENT')
    ser_con = pd.Series(ser_dlt * ser_coe, name='CONTRIBUTION')
    dtf_tmp = pd.concat([ser_dlt, ser_coe, ser_con], axis=1, verify_integrity=True)

    delta_abs = sum(abs(ser_con))

    dtf_tmp['ABS_CONTRIBUTION'] = abs(ser_con)
    dtf_tmp['PRC_ABS_CONTRIBUTION'] = dtf_tmp['ABS_CONTRIBUTION'] / delta_abs
    # order is crucial as we want to plot starting from the most important features
    dtf_tmp.sort_values(by='ABS_CONTRIBUTION', ascending=False, inplace=True)
    dtf_tmp['CUMULATIVE_PRC_ABS_CONTRIBUTION'] = round(dtf_tmp['PRC_ABS_CONTRIBUTION'].cumsum(), 2)

    dtf_tmp['FLG_VIZ'] = np.where(dtf_tmp['CUMULATIVE_PRC_ABS_CONTRIBUTION'] <= level_ths, 1, 0)
    # #73e2f7 tiffany
    # #e1c4ff lilac
    dtf_tmp['COLOR'] = np.where(dtf_tmp['CONTRIBUTION'] >= 0, '#73e2f7', '#e1c4ff')

    dtf_viz = dtf_tmp.loc[dtf_tmp['FLG_VIZ'] == 1].copy()
    dtf_viz.sort_values(by='ABS_CONTRIBUTION', ascending=True, inplace=True)

    if dtf_viz.shape[0] > maxnum_feat:
        dtf_viz = dtf_viz.iloc[-maxnum_feat:].copy()
        logging.debug(f"PLOT: CONTRIBUTION: REDUCED TO {maxnum_feat} FEATURES")

    if scale == 'log':
        dct_values = dtf_viz['CONTRIBUTION'].apply(lambda x: np.sign(x) * np.log1p(abs(x))).to_dict()
    elif scale == 'prc':
        dct_values = dtf_viz.apply(
            lambda x:
            round(np.sign(x['CONTRIBUTION']) * x['PRC_ABS_CONTRIBUTION'] * 100, 1),
            axis=1
        ).to_dict()
    else:
        dct_values = dtf_viz['CONTRIBUTION'].to_dict()
    colors = dtf_viz['COLOR'].to_list()

    fig, ax = plt.subplots(figsize=(25, 10))
    y_pos = np.arange(len(dct_values.keys()))
    bar_container = ax.barh(y_pos, dct_values.values())

    for i, (bar, color, value) in enumerate(zip(bar_container, colors, dct_values.values())):
        bar.set_color(color)
        bar.set_edgecolor('grey')
        bar.set_linewidth(0.5)
        # if value > 0:
        #     x_pos = bar.get_width() + x_pos_offset
        # else:
        #     x_pos = bar.get_width() - x_pos_offset
        # ax.text(
        #     x_pos,
        #     y_pos[i],
        #     round(abs(value), 2),
        #     ha='center',
        #     va='center',
        #     fontweight='bold'
        # )

    plt.yticks(y_pos, labels=dct_values.keys(), rotation=0, ha='right', fontsize=12)
    ax.set_xticklabels([])

    plt.title(
        f"EICER",
        y=1.15,
        fontweight='bold'
    )
    plt.suptitle(
        f"Comparing the item '{ser_h.name}' at position {rank_h} with the item '{ser_l.name}' at position {rank_l}.\n"
        f"Level of information: {level_detail}.\n"
        f"Scaling applied for the plot: {str(scale)}.",
        y=0.95,
    )

    # legend
    dct_item = {
        f'CONTRIBUTION TO ITEM {ser_h.name}': '#73e2f7',
        f'CONTRIBUTION TO ITEM {ser_l.name}': '#e1c4ff'
    }
    labels = list(dct_item.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=dct_item[lab]) for lab in labels]
    plt.legend(
        handles,
        labels,
        loc='best',
        fontsize=12,
    )

    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(right=0.85)

    str_name = 'RANK_COMPARISON'
    if scale:
        str_name = str_name + '_' + scale
    plt.savefig(str_path_out + str_name + '_' + str(ser_h.name) + '_' + str(ser_l.name) + '.png')

    return
