import pandas as pd

from typing import Union

from core.viz.viz_exp import explainer_delta_rank


def run_eicer_viz(
        dct_coef: dict,
        dtf_exp: pd.DataFrame,
        idx_h: Union[str, int],
        idx_l: Union[str, int],
        scale: str = None,
        level_detail: str = None,
) -> None:

    """
    Pre-process needed for the item evaluative contrastive graph.
    Input series are prepared and verified.
    :param dct_coef: dictionary with the coefficient of the logistic model.
    :param dtf_exp: dataframe with the feature values of the items to explode.
    :param idx_h: index of the item with higher rank.
    :param idx_l: index of the item with lower rank.
    :param scale: the scaling method applied to the features before plotting. Allowed values are None, log and prc (%).
    :param level_detail: the amount of information explaining the ranking difference that the user wants to visualize.
        Allowed values are LOW, MEDIUM, HIGH, ALL, each corresponding to a certain percentage.
    :return:
    """

    if idx_h not in dtf_exp.index:
        raise KeyError(f"index {idx_h} selected not in dtf_exp")
    if idx_l not in dtf_exp.index:
        raise KeyError(f"index {idx_l} selected not in dtf_exp")

    ser_h = dtf_exp.loc[idx_h].copy()
    ser_l = dtf_exp.loc[idx_l].copy()

    explainer_delta_rank(
        ser_h=ser_h,
        ser_l=ser_l,
        dct_coef=dct_coef,
        scale=scale,
        level_detail=level_detail,
        maxnum_feat=10,
    )

    return
