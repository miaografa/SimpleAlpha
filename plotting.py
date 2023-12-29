# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks

from functools import wraps
import utils
import performance as perf

DECIMAL_TO_BPS = 10000


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            color_palette = sns.color_palette('colorblind')
            with plotting_context(), axes_style(), color_palette:
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create alphalens default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """Create alphalens default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_information_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    print("Information Analysis")
    utils.print_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_ic_ts(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=2, lw=1.5, color='steelblue',ax=a)
        a.set(ylabel='IC', xlabel="")
        a.set_title(
            "{} Period Forward Return Information Coefficient (IC)"
            .format(period_num))
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax


def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sns.histplot(ic.replace(np.nan, 0.), kde=True, ax=a)
        a.set(title="%s Period IC" % period_num, xlabel='IC')
        a.set_xlim([-0.25, 0.25])
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sm.qqplot(ic.replace(np.nan, 0.).values, theoretical_dist, fit=True,
                  line='45', ax=a)
        a.set(title="{} Period IC {} Dist. Q-Q".format(
              period_num, dist_name),
              ylabel='Observed Quantile',
              xlabel='{} Distribution Quantile'.format(dist_name))
    return ax


def plot_ML_table(ml_record, baseline_name='baseline'):
    '''
    画出ML_record的表格，包括每个record的t检验。
    params: baseline_set: 基准特征集合的record name。
    '''
    ml_record = ml_record.copy()
    # 1. t检验：ml_summary_tabel
    if baseline_name not in ml_record.columns:
        baseline_name = ml_record.columns[0]  # 如果没有指定baseline，默认为第一个

    base_record = ml_record[baseline_name]
    target_df = ml_record.drop(baseline_name, axis=1)
    t_ind_list = []
    p_ind_list = []
    t_rel_list = []
    p_rel_list = []
    for record_name in target_df.columns:
        temp_record = target_df[record_name]
        # t检验
        t_stat, p_value = stats.ttest_ind(base_record, temp_record)  # 均值是否一致, H0: 没有差异， H1：存在差异。
        t_ind_list.append(t_stat)
        p_ind_list.append(p_value)
        t_stat, p_value = stats.ttest_rel(base_record, temp_record)  # 差值是否存在, H0:差值为0
        t_rel_list.append(t_stat)
        p_rel_list.append(p_value)

    ml_summary_tabel = pd.DataFrame({
        'record_name': target_df.columns,
        't_ind': t_ind_list,
        'p_ind': p_ind_list,
        't_rel': t_rel_list,
        'p_rel': p_rel_list
    })
    ml_summary_tabel.set_index('record_name', drop=True, inplace=True)

    # 2. ml_record
    mean_record = ml_record.mean()
    mean_record.name = 'mean'
    std_record = ml_record.std()
    std_record.name = 'std'
    ml_record = ml_record.append(mean_record)
    ml_record = ml_record.append(std_record)

    print("Machine Learning records")
    utils.print_table(ml_record.apply(lambda x: x.round(4)).T)
    print("Machine Learning T-test Analysis")
    utils.print_table(ml_summary_tabel.apply(lambda x: x.round(4)))
    return



def plot_ML_records(ml_record):
    ml_record.iplot('bar')