__all__ = [
    'plot_corr_matrix',
    'plot_linear_fit',
    'add_text_adjust',
    'swarmplot',
    'boxplot',
    'plot_kaplan_pval',
]

import itertools

import adjustText
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
import lifelines

from .utils import get_cross_cor, stats_to_str, calc_pvals_sign


def plot_corr_matrix(df=None, names=None, dtr_var=None, abs_=False, save=None, cmap=None, vmax=1., vmin=None,
                     metric=None, corr=None, scale_size=1., figsize=None, dpi=300, **kwargs):
    if corr is None:
        if isinstance(df, np.ndarray):
            names = np.arange(df.shape[1]) if names is None else names
            df = pd.DataFrame(data=df, columns=names)
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            raise ValueError('Datatype is not understood %s' % type(df))

        names = df.columns if names is None else names
        names = np.asarray(names)
        if dtr_var is not None:
            dtr = df[dtr_var].values
            m = np.isfinite(dtr)
            for g_ in names:
                y = df[g_].values
                m *= np.isfinite(y)
                df[g_] = y - np.poly1d(np.polyfit(dtr[m], y[m], 1))(dtr)

        corr = get_cross_cor(df, names, abs_, metric)
    else:
        names = np.arange(corr.shape[1]) if names is None else names

    if cmap is None:
        cmap = plt.get_cmap('coolwarm')

    if figsize is None:
        figsize = (max(5, len(names) * 0.3) * scale_size, max(5 * 0.8, 0.8 * len(names) * 0.3) * scale_size)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    text_pad_h = 0.12
    matrix_w = 0.6
    matrix_h = 0.81
    dendro_w = 0.1
    cbar_w = 0.05
    width = text_pad_h * 1.8 + matrix_w + dendro_w + cbar_w
    scale_w = 1 / width

    axmatrix = fig.add_axes([text_pad_h*scale_w, 0.02, matrix_w*scale_w, matrix_h])
    axcbar = fig.add_axes([(text_pad_h+matrix_w+dendro_w)*scale_w, 0.02, cbar_w*scale_w, matrix_h])
    axdendro = fig.add_axes([(text_pad_h+matrix_w)*scale_w, 0.02, dendro_w*scale_w, matrix_h])

    method = kwargs.pop('method', 'centroid')
    optimal_ordering = kwargs.pop('optimal_ordering', True)
    color_threshold = kwargs.pop('color_threshold', 0)
    metric = kwargs.pop('metric', 'euclidean')
    Y = sch.linkage(corr, method=method, optimal_ordering=optimal_ordering, metric=metric)
    Z = sch.dendrogram(Y, orientation='right', color_threshold=color_threshold, above_threshold_color='grey')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    # Plot distance matrix.
    index = Z['leaves']
    corr = corr[index, :]
    corr = corr[:, index]
    if vmin is None:
        if abs_:
            vmin = 0
        else:
            vmin = -1
    mtx = axmatrix.matshow(corr, aspect='auto', origin='lower', cmap=cmap, vmax=vmax, vmin=vmin)

    # Horizontal ticks
    rotation = kwargs.pop('rotation', 80)
    fontsize = kwargs.pop('fontsize', 12)

    axmatrix.minorticks_off()
    axmatrix.set_xticks(np.arange(len(names)))
    axmatrix.set_xticklabels(names[index], rotation=rotation, fontsize=fontsize, ha="left")
    axmatrix.xaxis.set_label_position('top')

    # Vertical ticks
    axmatrix.set_yticks(np.arange(len(names)))
    axmatrix.set_yticklabels(names[index], fontsize=fontsize)
    axmatrix.grid(False)

    cbar_label = kwargs.pop('label', '')
    cb = plt.colorbar(mtx, pad=0.05, cax=axcbar)
    cb.set_label(cbar_label, fontsize=fontsize)
    axcbar.tick_params(labelsize=fontsize)
    # if save is not None:
        # plt.savefig(save, bbox_inches='tight')
    return fig, corr, names[index]


def plot_linear_fit(x, y, x_max=None, x_min=None, ax=None, text='', c='k', fmt='cor', line=True, **fmt_kwargs):
    ax = plt.gca() if ax is None else ax
    x = np.array(x)
    y = np.array(y)
    m = np.isfinite(x) * np.isfinite(y)
    x_max = x[m].max() if x_max is None else x_max
    x_min = x[m].min() if x_min is None else x_min
    x_min += 1e-5
    x_ = np.linspace(x_min, x_max, 10)
    lr = scipy.stats.linregress(x[m], y[m])
    lbl1 = stats_to_str(x, y, fmt, text, **fmt_kwargs)
    if line:
        ax.plot(x_, x_ * lr[0] + lr[1], '--', c=c, lw=3, label=lbl1)
    else:
        ax.scatter([], [], label=lbl1)
    return ax


def add_text_adjust(x, y, text, ax=None, fs=12, **kwargs):
    ax = plt.gca() if ax is None else ax
    text_anno = []
    for tup in zip(x, y, text):
        text_anno.append(ax.text(*tup, fontsize=fs))

    default_props = dict(arrowprops=dict(arrowstyle="->", color='k', lw=1.2), force_text=0.1, force_points=2.5, )
    default_props.update(kwargs)

    adjustText.adjust_text(text_anno, ax=ax, **default_props)
    return ax


def swarmplot(df, x_key="group", y_key='bio', order=None, ax=None,
              colors=None, title='', xlabel='', ylabel=None, pval_func=None,
              sign_=0.05, markersize=10, fontsize=12, rotation=0):
    # if colors is not None:
    #     plt.rc('axes', prop_cycle=cycler('color', plt_colors))
    ax = plt.gca() if ax is None else ax
    order = df[x_key].unique().astype(np.unicode_) if order is None else order

    sns.swarmplot(x=x_key, y=y_key, hue=colors, data=df, size=markersize, order=order, ax=ax)
    ax.plot([-0.5, len(order) - 0.5], [df[y_key].median(), df[y_key].median()], '--', color='grey')

    med_width = 0.1
    for i in range(len(order)):
        ax.plot([i - med_width, i + med_width],
                [df[df[x_key] == order[i]][y_key].median(), df[df[x_key] == order[i]][y_key].median()], '-', color='k',
                lw=2.5, zorder=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    xticks = ax.get_xticklabels()
    ax.set_xticklabels(xticks, rotation=rotation)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    def round_1digit_alt(x):
        x = 1.0E-33 if x == 0 else x
        ndig = int(-np.log10(x))
        if ndig < 5:
            fmt = "p=%%.%df" % (ndig + 1)
        else:
            fmt = "p=%.2e"
        return fmt % x

    if pval_func is None:
        pval_func = lambda x, y: scipy.stats.mannwhitneyu(x, y)[1]
    else:
        if not callable(pval_func):
            raise ValueError("pval_func should be callable")

    box_ind = np.arange(len(order), dtype=int)

    min_vals = df.groupby(x_key).min().loc[order][y_key].values
    max_vals = df.groupby(x_key).max().loc[order][y_key].values  # np.argsort()
    diff = max_vals.max() - min_vals.min()

    comb = np.array(list(itertools.combinations(box_ind, 2)))
    diff_comb = np.diff(comb, axis=1)

    pval_sign = 0
    for c in np.argsort(diff_comb.flatten()):
        i, j = comb[c]
        gr1 = df[df[x_key] == order[i]][y_key].dropna().values
        gr2 = df[df[x_key] == order[j]][y_key].dropna().values
        pval_ = pval_func(gr1, gr2)
        #         title += '$p_{%d,%d}=$%.1e\t'%(i+1, j+1, pval)
        if pval_ < sign_:
            pval_shift = (pval_sign + 1) * diff * 0.008 * fontsize  # 0.35
            draw_pvals_sign(round_1digit_alt(pval_), [i, j], max_vals.max() + pval_shift, diff,  ax, fs=fontsize)
            pval_sign += 1
    return ax


def draw_pvals_sign(text, x, y, v_size=None, ax=None, fs=10):
    """
    Draw cap with text above

    Parameters
    ----------
    text : basestring
        text to add
    x : tuple, size 2
        X-coordinates of cap, left and right position
    y : tuple, size 2
        y-coordinates of the cap, bottom and top elements, y[1] = y + h
    ax : matplotlib.pyplot.Axes object
        axes for plotting
    fs : int
        fontsize
    v_size: float
        vertical size of the image
    """

    x_mid = (x[0] + x[1]) * 0.5
    if v_size is None:
        v_size = min(y, 1)
    h = v_size * 0.025  # cap size
    ax.plot([x[0], x[0], x[1], x[1]], [y, y + h, y + h, y], lw=1.5, c='grey')
    ax.text(x_mid, y + h, text, ha='center', va='bottom', color='k', fontsize=fs)


def boxplot(data, labels=None, names=None, showpvals=False, pval_func=None, alpha=0.05, color=None,
            widths=0.6, cmap='viridis', fontsize=14, show_median=True, show_mean=False,
            medianshift=None, ax=None, **kwargs):
    """ Plot boxplots for given samples or provided quartiles

    Parameters
    ----------
    data : array_like
        `N` arrays of sample observations assumed, sample sizes can be different,
        or 1D array
    labels : array_like, type of string, shape (n_groups,)
        list of labels for each box
        or If data 1D array, is 1D array
    names : array_like, type of string
        list of box names which is used for plotting, can be used for forsing desired order of
        labels or hiding some labels. Is used only if labels is 1D array
    color : dict, optional
        maps color to each box
    showpvals : bool, optional
        If `True` will add p-values in title, default is `False`
    alpha : float, optional
        Significance level for p-values, if p-value above `alpha` it will not be shown
    pval_func : callable
        A function, which accepts two arrays of one-dimensional samples and returns p-value on difference in means.
        Default is `scipy.stats.mannwhitneyu`
    cmap : str, optional
        name of colormap, default is viridis
    widths : float, optional
        Sets box width , default is 0.6
    show_median : bool, optional
        show median values on plot
    show_mean : bool, optional
        show mean instead of median (for internal use in Gero only)
    fontsize : int, optional
        font size used for medians
    medianshift : int, optional
        shifts mendian text vertically
    ax : matplotlib.pyplot.Axes object
        axes for plotting

    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        axes for plotting

    Examples
    --------
    >>> boxplot([np.random.randn(20)*0.2 + 1, np.random.randn(20)*0.5 - 1, np.random.randn(20)*2 - 1], showpvals=True)
    >>> plt.show()
    """

    nlabel = len(data)
    labels = np.arange(1, nlabel + 1).astype('unicode') \
        if labels is None else np.asarray(labels, dtype=str)
    assert len(data) == len(labels), 'Shapes data/label mismatch {} {}'.format(len(data),
                                                                               len(labels))
    labels_unique = list(set(labels))
    labels_unique = np.sort(labels_unique)
    assert len(labels_unique) <= 50, "Too many labels"

    if len(labels_unique) != len(labels):
        if names is not None:
            names = np.asarray(names, dtype=str)
            assert len(np.intersect1d(names, labels_unique)) == len(names), \
                f"Some names in {names} was not found in labels {labels_unique}"
            labels_unique = names
        data = [np.asarray(data[labels == uval]) for uval in labels_unique]
        labels = labels_unique

    # Calculate p-vals
    pvals_list = None
    if showpvals:
        pvals_list = calc_pvals_sign(data, 1, pval_func, alpha)

    # Calculate quartiles for plotting
    quartiles = []
    for d in data:
        quartile_ = np.nanpercentile(d, [5, 25, 50, 75, 95])
        if show_mean:
            quartile_[2] = np.nanmean(d)
        quartiles.append(quartile_)

    quartiles = np.array(quartiles)
    ax = boxplot_q(quartiles, labels, pvals=pvals_list, color=color, cmap=cmap, widths=widths,
                   fontsize=fontsize, medianshift=medianshift, ax=ax,
                   show_median=show_median, **kwargs)

    return ax, quartiles, labels


def boxplot_q(quartiles, labels, color=None, showfliers=False, pvals=None,
              cmap='viridis', widths=0.6, fontsize=14, medianshift=None, ax=None,
              show_median=True, **kwargs):
    """ Plot boxplots from quartiles

    Parameters
    ----------
    quartiles : array_like
        `N` arrays of quartiles [Q1,Q2,Q3] corresponding to 25th percentile, 50th percentile and 75th percentile
        or `N` arrays of quartiles [LO, Q1,Q2,Q3, HI]
    labels : array_like, type of string, shape (n_groups,)
        list of labels for each box
    color : dict, optional
        maps color to each box
    showfliers : bool, optional
        If `True` will show outliers
    cmap : str, optional
        name of colormap, default is viridis
    widths : float, optional
        Sets box width , default is 0.6
    medianprops : dict, optional
        set up median style
    show_median : bool, optional
        show median values on plot
    capprops : dict, optional
        set up cap style
    whiskerprops : dict, optional
        set up whisker style
    fontsize : int, optional
        font size used for medians
    medianshift : int, optional
        shifts mendian text vertically
    ax : matplotlib.pyplot.Axes object
        axes for plotting

    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        axes for plotting
    """

    def _set_box_style(bpl, colors, ax_, show_meds=True, medianshift=0, fs=None):
        """ Set colors, medians, etc for a single boxplot """
        for patch, color_ in zip(bpl['boxes'], colors):
            patch.set_facecolor(color_)
            patch.set_alpha(0.2)
        for patch, color_ in zip(bpl['fliers'], colors):
            patch.set_markerfacecolor(color_)
        for patch, color_ in zip(bpl['medians'], colors):
            patch.set_color(color_)

        if show_meds:
            i = 1
            for m in bpl['medians']:
                xm, ym = m.get_data()
                ax_.annotate('%5.2f' % ym[0],
                             xy=(xm[0] + (xm[1] - xm[0]) * 0.2, ym[0] + medianshift), fontsize=fs)
                i += 1

    assert len(quartiles) == len(labels), "Mismatch of quartiles and label sizes {} and {}".format(
        len(quartiles),
        len(labels))
    quartiles = np.asarray(quartiles)
    if quartiles.shape[1] != 3 and quartiles.shape[1] != 5:
        raise ValueError(
            "Data should be shape of (N, 3) or (N,5). Your shape is (%s)" % (
                ','.join(map(str, quartiles.shape))))

    nlabel = len(quartiles)

    if ax is None:
        ax = plt.gca()

    # Set colors
    if color is None and cmap is not None:  # generate n colors from cmap
        color = colors_from_cmap(nlabel, cmap=cmap)
    elif color is None and cmap is None:  # use default colors in matplotlib
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color = prop_cycle.by_key()['color']
        color = color * int(np.ceil(nlabel / len(color)))
        color = color[:nlabel]

    medianprops = kwargs.get('medianprops', dict(linestyle='-', linewidth=8))
    capprops = kwargs.get('capprops', dict(linewidth=0))
    whiskerprops = kwargs.get('whiskerprops', dict(linewidth=2))

    val_min = 1e-8
    val_max = -1e-8
    for b in quartiles:
        val_min = min(val_min, np.asarray(b).min())
        val_max = max(val_max, np.asarray(b).max())
    diff = val_max - val_min

    if medianshift is None:
        medianshift = (val_max - val_min) * 0.035

    boxplots = ax.boxplot([[-2, -1, 0, 1, 2]] * nlabel, labels=labels, patch_artist=True,
                          medianprops=medianprops,
                          capprops=capprops, whiskerprops=whiskerprops, showfliers=showfliers,
                          widths=widths)
    y_lo = np.inf
    y_hi = -np.inf

    for box_no, d in enumerate(quartiles):
        if d.shape[0] == 3:
            q1, q2, q3 = d
            iqr = q3 - q1
            q1_start = q1 - 1.5 * iqr
            q4_end = q3 + 1.5 * iqr

        else:
            q1_start, q1, q2, q3, q4_end = d

        y_lo = np.min([y_lo, q1_start])
        y_hi = np.max([y_hi, q4_end])

        # Lower cap
        boxplots['caps'][2 * box_no].set_ydata([q1_start, q1_start])

        # Lower whiskers
        boxplots['whiskers'][2 * box_no].set_ydata([q1, q1_start])

        # Higher cap
        boxplots['caps'][2 * box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        boxplots['whiskers'][2 * box_no + 1].set_ydata([q3, q4_end])

        # Box
        boxplots['boxes'][box_no]._path._vertices[:-1, 1] = [q1, q1, q3, q3, q1]

        # Medians
        boxplots['medians'][box_no].set_ydata([q2, q2])

    _set_box_style(boxplots, color, ax, show_median, medianshift, fontsize)
    ax.set_xticklabels(labels, fontsize=fontsize)
    if pvals is not None:
        assert isinstance(pvals, list)
        for pval in pvals:
            draw_pvals_sign(*pval, ax=ax, fs=fontsize)
            y_hi = max(y_hi, np.max(pval[2]))

    ax.set_ylim(y_lo - diff * 0.035, y_hi + diff * 0.035)
    return ax


def colors_from_cmap(n=5, cmap='viridis_r', drop_edges=None):
    """ Generate discrete colors from colormap

    Example:
    colors = [cm.jet_r(i / float(n+2)) for i in range(n+2)][1:-1]
    """
    if drop_edges is None:
        drop_edges = (cmap if isinstance(cmap, str) else cmap.name) in {"jet", "jet_r", "viridis", "viridis_r", "plasma", "plasma_r"}
    cmap = plt.get_cmap(cmap)
    if drop_edges:
        return [cmap(i) for i in np.linspace(0, 1, n + 2)[1:-1]]
    return [cmap(i) for i in np.linspace(0, 1, n)]


def plot_kaplan_pval(df, ax=None):
    df_ = df.dropna(subset=['mort_age'])
    tte = df_['mort_age'].values
    ev = np.ones_like(tte, dtype=int)
    lbl = df_['group'].str.strip('W').values
    m = lbl == 'RD'
    res = lifelines.statistics.logrank_test(tte[m], tte[~m])
    if ax is None:
        ax = plt.gca()
    kaplan_plot(tte, ev, lbl, flatten=True, alpha=0., ax=ax, colors=plt.get_cmap('tab10').colors[:4])
    ax.annotate('p=%.3f' % res.p_value, (tte.max() * 0.85, 0.5), fontsize=14)
    ax.set_xlim(50, np.max(tte) + 10)
    ax.set_ylabel('Survival')
    ax.set_xlabel('Mortality age, weeks')
    return ax


def kaplan_plot(tte, events, labels=None, flatten=False, colors=None, cmap='viridis',
                lw=5, mew=1.2, ms=6, fs=None, alpha=0.2, ax=None):
    """ Make Kaplan-Meyer plot for given events and time-to-event data.

    Parameters
    ----------
    tte :  1D array or list of arrays, array_like
        time-to-event arrays, could be different sizes.
        If flatten parameter is True, tte is 1D array
    events : 1D array or list of arrays, array_like
        events arrays, exactly match dimensions of tte
        If flatten parameter is True, tte is 1D array
    labels : array_like, type of string
        list of labels for each line, if flatten parameter is False
        or label for the each sample if flatten parameter is True
    colors : dict, optional
        maps color to each box
    cmap : colormap, optional
        colormap, default is viridis
    lw : int, optional
        line width, default 5
    mew : float, optional
        censored marker width, default 1.2
    ms : int, optional
        censored marker size, default 6
    fs : int, optional
        legend font size, default None
    alpha : float, optional
        0.0 transparent through 1.0 opaque, default is 1.0
    ax : matplotlib.pyplot.Axes object
        axes for plotting

    Returns
    -------
    ax : matplotlib.pyplot.Axes object
        axes for plotting

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> kaplan_plot([np.random.randn(100)+6., np.random.randn(150)+6.],
                        [np.random.randint(0,2,100), np.random.randint(0,2,150)],
                        labels=['Rand1', 'Rand2'],ax=ax)
    >>> fig.set_size_inches(10,10)
    >>> plt.show()

    >>> kaplan_plot(np.random.randn(200)+2., np.random.randint(0,2,200),
                        labels=np.random.randint(0,4,200), flatten=True, ax=ax)
    >>> fig.set_size_inches(10,10)
    >>> plt.show()
    """

    if flatten is True:
        assert len(tte) == len(events) == len(labels)
        labels_unique = list(set(labels))
        tte = [list(tte[labels == uval]) for uval in labels_unique]
        events = [list(events[labels == uval]) for uval in labels_unique]
        labels = labels_unique

    assert len(tte) == len(events)
    nlines = len(tte)
    labels = np.arange(nlines).astype(str) if labels is None else np.char.asarray(labels)
    assert len(labels) == nlines

    if isinstance(colors, type(None)):
        colors = colors_from_cmap(max(4, nlines), cmap=cmap, drop_edges=False)
        colors = np.roll(colors, -1, axis=0)

    if ax is None: ax = plt.gca()

    kmf = lifelines.KaplanMeierFitter()
    for l in range(nlines):
        kmf.fit(tte[l], event_observed=events[l])
        y = kmf.survival_function_.values
        x = kmf.survival_function_.index.values
        cens = kmf.event_table['censored'].values

        ax.plot(x, y, lw=lw, label=labels[l], color=colors[l], drawstyle='steps-post')
        ax.plot(x[cens > 0], y[cens > 0], 'x', c='k', mew=mew, ms=ms)
        ax.fill_between(x, kmf.confidence_interval_.values[:, 0],
                        kmf.confidence_interval_.values[:, 1], color=colors[l], alpha=alpha)
    ax.legend(loc=0, fontsize=fs)
    del kmf
    return ax
