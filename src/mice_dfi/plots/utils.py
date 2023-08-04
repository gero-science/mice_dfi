__all__ = [
    'stats_to_str',
    'features_rename_dict',
    'strain_rename_dict',
    'rename_features',
    'get_pairs_longitudinal',
    'linear_detrending',
    'medmad',
    'calc_delta',
]

import itertools
from typing import Union, Optional

import numpy as np
import scipy.stats
import sklearn


def stats_to_str(x, y, stats='cor', text='', pcap=1e-50, pdig=5, add_star=False):
    """ Calculate statistic test and prepare pretty label from the results """
    x = np.array(x)
    y = np.array(y)
    m = np.isfinite(x) * np.isfinite(y)
    pval = 1.
    if stats in ['cor', 'spear']:
        if stats == 'cor':
            r, pval = scipy.stats.pearsonr(x[m], y[m])
        elif stats == 'spear':
            r, pval = scipy.stats.spearmanr(x[m], y[m])
        else:
            raise Exception
        lbl1 = '{:s} r={:.2f}'.format(text, r)
    elif stats == 'exp':
        r, pval = scipy.stats.pearsonr(x[m], y[m])
        lbl1 = '{:s} R={:.2f}'.format(text, r)
    elif stats == 'expN':
        stat = scipy.stats.pearsonr(x[m], y[m])
        lbl1 = '{:s} R={:.2f}, N={:d}'.format(text, stat[0], sum(m))
        pval = stat[1]
    elif stats == 'lin':
        lr = scipy.stats.linregress(x[m], y[m])
        lbl1 = '{:s} (x0={:.3f}, t={:.4f})'.format(text, lr[1], lr[0])
        pval = lr[3]
    elif stats == 'r2':
        lr = scipy.stats.linregress(x[m], y[m])
        lbl1 = '{:s} $R^2$={:.2f}'.format(text, lr[2] ** 2)
        pval = lr[3]
    elif stats == 'empty' or stats is None:
        lbl1 = '{:s}'.format(text)
    else:
        raise ValueError(f'Unknown {stats=}, please use following  options [cor, spear, lin, exp, expN, r2, empty]')

    if stats != 'empty':
        if add_star:
            if pval < pcap:
                lbl1 += '*'
        else:
            if pval < pcap:
                lbl1 += ', p<{:g}'.format(pcap)
            elif pval < np.power(10., -pdig):
                lbl1 += ', p={:1.0e}'.format(pval)
            else:
                fmt = ', p={:.' + str(pdig) + 'f}'
                lbl1 += fmt.format(np.round(pval, pdig))
    return lbl1


def get_cross_cor(df, names, abs_=False, metric=None):
    """ Calculate cross-correlation matrix (ignores NaN and inf)"""
    if metric is None or metric == 'corr':
        f_metric = lambda x, y: np.corrcoef(x, y)[0, 1]
    elif metric == 'cosine':
        f_metric = lambda x, y: 1 - scipy.spatial.distance.cosine(x, y)
    else:
        raise ValueError(f'{metric}= is not recognized')

    corr = np.zeros((len(names), len(names)))
    for i, g1 in enumerate(names):
        for j, g2 in enumerate(names):
            x1 = df[g1].values
            x2 = df[g2].values
            m = np.isfinite(x1) * np.isfinite(x2)
            corr[i, j] = f_metric(x1[m], x2[m])
            if abs_:
                corr[i, j] = np.fabs(corr[i, j])
    return corr


strain_rename_dict = {
    '129s1/svimj': '129S1/SvImJ',
    'a/j': 'A/J',
    'balb/cbyj': 'BALB/cByJ',
    'btbr t+ tf/j': 'BTBR T$^\mathrm{+}$tf/J',
    'bub/bnj': 'BUB/BnJ',
    'c3h/hej': 'C3H/HeJ',
    'c57bl/10j': 'C57BL/10J',
    'c57bl/6j': 'C57BL/6J',
    'c57blks/j': 'C57BLKS/J',
    'c57br/cdj': 'C57BR/cdJ',
    'c57l/j': 'C57L/J',
    'cast/eij': 'CAST/EiJ',
    'cba/j': 'CBA/J',
    'dba/2j': 'DBA/2J',
    'fvb/nj': 'FVB/NJ',
    'kk/hlj': 'KK/HlJ',
    'lp/j': 'LP/J',
    'mrl/mpj': 'MRL/MpJ',
    'nod.b10sn-h2<b>/j': 'NOD.B10Sn-H2$^b$/J',
    'non/shiltj': 'NON/ShiLtJ',
    'nzo/hlltj': 'NZO/HlLtJ',
    'nzw/lacj': 'NZW/LacJ',
    'p/j': 'P/J',
    'pl/j': 'PL/J',
    'pwd/phj': 'PWD/PhJ',
    'riiis/j': 'RIIIS/J',
    'sjl/j': 'SJL/J',
    'sm/j': 'SM/J',
    'swr/j': 'SWR/J',
    'wsb/eij': 'WSB/EiJ'}

features_rename_dict = {
    'bw': 'BW',
    'crp': 'CRP',
    'dia': 'Dia',
    'eo %': 'EO%',
    'eo (k/ul)': 'EO',
    'flow': 'Flow',
    'glu': 'GLU',
    'gr %': 'GR%',
    'gr (k/ul)': 'GR',
    'gs': 'GS',
    'hb (g/dl)': 'HB',
    'hct %': 'HCT',
    'hr': 'HR',
    'ins': 'INS',
    'kc': 'KC',
    'ly %': 'LY%',
    'ly (k/ul)': 'LY',
    'mch (pg)': 'MCH',
    'mchc (g/dl)': 'MCHC',
    'mcv(fl)': 'MCV',
    'mean': 'Mean',
    'mo %': 'MO%',
    'mo (k/ul)': 'MO',
    'mpv (fl)': 'MPV',
    'ne %': 'NE%',
    'ne (k/ul)': 'NE',
    'plt (k/ul)': 'PLT',
    'rbc (m/ul)': 'RBC',
    'rdw %': 'RDW%',
    'sys': 'Sys',
    'tg': 'TG',
    'volume': 'Volume',
    'wbc (k/ul)': 'WBC',
    'igf1': 'IGF1',
}


def rename_features(a):
    a = np.copy(a)
    for i, name in enumerate(a):
        if name in features_rename_dict:
            a[i] = features_rename_dict[name]
    return a


def get_pairs_longitudinal(df, params, groups=None, age_min=0, age_max=200, dt=1., dt_tol=0,
                               normalize=True, unique_age=True, drop_age=None, adjust_age=False,
                               adjust_sample=False, max_pairs=None, index=False,
                               timelag=False):
    """ Extract all pair from longitudinal dataset with time between measurements `dt`

    Parameters
    ----------
    unique_age : bool, default True
        If True assume discrete age values (like mouse dataset), else continuous age
    drop_age : list, default None
        Specific ages to exclude
    adjust_age : bool, default False
        Do bin-wise detrending for unique_age mode
    index : bool, default False
        Return indices for Dataframe for selected pairs

    """

    def np_append(a, b):
        """ Concatenate two arrays """
        return b if a is None else np.concatenate((a, b))

    df = df.copy()
    filter_age = [] if drop_age is None else drop_age

    df = df[(df.age >= age_min) & (df.age <= age_max) & ~(df.age.isin(filter_age))]  # filter age

    if groups is not None:
        df = df[df.group.isin(groups)]

    if adjust_sample is True:
        if normalize is True:
            raise ValueError('You should not use normalize=True with adjust_sample=True')

    if isinstance(params, str):
        params = (params,)

    if adjust_age and not unique_age:
        print("adjust age for non-unique labels")
        for gr_, df_ in df.groupby('age'):
            df.loc[df_.index, params] -= df_[params].median()

    X_all = None
    Y_all = None
    ind_X = None
    ind_Y = None

    # Index prop
    index_name = df.index.name
    if index_name is None or len(index_name) == 0:
        index_name = "_ID_"
        df.index.name = index_name

    if unique_age:
        available_ages = df.age.unique()
        available_ages = np.sort(available_ages)
        age_diff = np.diff(available_ages)

        valid_pairs = abs(age_diff - dt) <= dt_tol
        if not valid_pairs.any():
            raise ValueError('Failed to find pairs, change `dt` or `dt_tol` %s' % ((age_diff),))
        age_0 = available_ages[:-1][valid_pairs]
        age_1 = available_ages[1:][valid_pairs]
        for a1, a2 in zip(age_0, age_1):
            df1 = df[df.age == a1].set_index('uid', append=True).reset_index(level=0)
            df2 = df[df.age == a2].set_index('uid', append=True).reset_index(level=0)
            ind_common = df1.index.intersection(df2.index)

            X = df1.loc[ind_common, params].values  # ML.get_X_from_df(df1.loc[ind_common], norms, params, normalize)
            Y = df2.loc[ind_common, params].values  # ML.get_X_from_df(df2.loc[ind_common], norms, params_Y, normalize)
            m = np.isfinite(X).all(axis=1) * np.isfinite(Y).all(axis=1)
            X = X[m]
            Y = Y[m]
            if adjust_age:
                X -= X.mean(axis=0)
                Y -= Y.mean(axis=0)
            X_all = np_append(X_all, X)
            Y_all = np_append(Y_all, Y)
            ind_X = np_append(ind_X, df1.loc[ind_common, index_name][m])
            ind_Y = np_append(ind_Y, df2.loc[ind_common, index_name][m])
    else:

        if adjust_sample:
            means = df.groupby('uid')[params].mean()
            df = df.reset_index().set_index(['uid', 'age'])
            df[params] -= means
            df = df.reset_index().set_index(index_name)

        df = df.sort_values(['uid', 'age'])
        df_uid = df.groupby('uid')

        for uid_, df_ in df_uid:
            df_ = df_.copy()
            assert ~df_.age.duplicated().any(), "No duplicated ages allowed for uid %s" % uid_
            if len(df_) == 1:
                continue

            available_ages = df_.age.values
            finite_mask = ~df_[params].isnull().any(axis=1).values
            available_ages = available_ages[finite_mask]
            pairs = np.asarray(list(itertools.combinations(available_ages, 2)))
            mask = (np.fabs(np.diff(pairs) - dt) <= dt_tol).flatten()

            if sum(mask) == 0:
                continue

            good_pairs = pairs[mask]

            if len(good_pairs) == 0:
                continue

            # Filter closed pairs
            closed_pairs = []
            dt_diff_scale = 0.25
            if len(good_pairs) >= 1:
                for i, pair in enumerate(good_pairs):
                    if i == 0:
                        t0 = pair[0]
                        continue
                    if pair[0] - t0 < dt * dt_diff_scale:
                        closed_pairs.append(i)
                    else:
                        t0 = pair[0]

            good_pairs = good_pairs.tolist()
            for i in sorted(closed_pairs, reverse=True):
                del good_pairs[i]
            good_pairs = np.asarray(good_pairs)

            # Randomly select max_pairs from available
            if max_pairs is not None and len(good_pairs) > max_pairs:
                selected = np.random.choice(np.arange(len(good_pairs), dtype=int), size=max_pairs, replace=False)
                good_pairs = good_pairs[selected]

            X = df_.set_index('age').loc[good_pairs[:, 0], params].values
            Y = df_.set_index('age').loc[good_pairs[:, 1], params].values
            X_all = np_append(X_all, X)
            Y_all = np_append(Y_all, Y)
            ind_X = np_append(ind_X, df_.reset_index().set_index('age').loc[good_pairs[:, 0], index_name])
            ind_Y = np_append(ind_Y, df_.reset_index().set_index('age').loc[good_pairs[:, 1], index_name])

    res = [X_all, Y_all]
    if index:
        res += [[ind_X, ind_Y]]
    if timelag:
        res += [np.copy(df.loc[ind_Y, 'age'].values - df.loc[ind_X, 'age'].values)]
    return res


def linear_detrending(df, features, key_dtr: Union[str, list] = 'age', how: str = 'lin', inplace: bool =False,
                      age_bins: Union[list, np.ndarray] = None, huber_eps: float = 2.0, deg: int = 1,
                      shift_mean: bool = False):
    """

    Parameters
    ----------
    df: DataFrame
    features: array-like, dtype str
        columns name which should be detrended
    key_dtr: str
        column name with variable
    how: str
        Defines method for detrending: 'lin', 'huber', 'binwise', 'groupby'
    inplace: bool
        if True replace old values with detrended ones
    age_bins: array-like
        age bins for `binwise` method
    huber_eps: float
        epsilon in huber regression (controls the outlier threshold)
    deg: int
        polynomial degree in `lin` method
    shift_mean : bool, optional
        if True, after detrending shifts dataset back to global mean values

    Returns
    -------
    df : pandas.Dataframe
    """
    if how not in ['lin', 'huber', 'binwise', 'groupby']:
        raise ValueError('Parameter how=%s is not understood, try {lin, huber, binwise, groupby}' % how,)
    df = df.copy()

    if shift_mean:
        means_ = df[features].dropna(subset=features).mean(axis=0).values
        print(means_)
    else:
        means_ = np.zeros(len(features))

    dtr = None
    isfinite_mask = None
    # Detrended variable is finite
    if how != 'groupby':
        dtr = df[key_dtr].values.copy()
        isfinite_mask = np.isfinite(dtr)

    bins = None
    if how == 'binwise':
        if age_bins is None:
            raise ValueError("Initialize age bins")
        bins = np.digitize(dtr, age_bins)

    for g_ in features:

        if inplace:
            key = g_
        else:
            key = g_ + '_dtr'
            df[key] = np.NaN

        if how == 'groupby':
            df[key] = df[g_].values.copy()
            for gr_, df_ in df.groupby(key_dtr):
                df.loc[df_.index, key] -= np.nanmean(df_[g_].values)
        else:
            y = df[g_].values.copy()
            isfinite_mask *= np.isfinite(y)

            if how == 'lin':
                df[key] = y - np.poly1d(np.polyfit(dtr[isfinite_mask], y[isfinite_mask], deg))(dtr)
            elif how == 'huber':
                huber = sklearn.linear_model.HuberRegressor(fit_intercept=True, alpha=0.0,
                                                            max_iter=100, epsilon=huber_eps)
                if deg > 1:
                    Xfit = np.copy(dtr)
                    for i in range(1, deg):
                        Xfit = np.column_stack((Xfit, dtr**(i + 1)))
                else:
                    Xfit = dtr.reshape(-1, 1)
                huber.fit(Xfit[isfinite_mask], y[isfinite_mask])
                df[key] = np.NaN
                df.loc[isfinite_mask, key] = y[isfinite_mask] - huber.predict(Xfit[isfinite_mask])
            elif how == 'binwise':
                for uval in np.unique(bins):
                    muval = bins == uval
                    df.loc[muval, key] = y[muval] - np.nanmean(y[muval])

    df[features] += means_
    return df


def medmad(x, threshold=3):
    """ Find outliers using Median Absolute Deviation  """
    med_x = np.nanmedian(x)
    mad_x = np.nanmedian(np.fabs(x - med_x))
    mask = 1.4826 * np.fabs(x - med_x) / mad_x > threshold
    return mask


def calc_delta(df, feature, shift=1, key_id='uid', key_time='age'):
    assert not df.index.has_duplicates, "df Index has duplicates, please fix this"

    if df.duplicated(subset=[key_id, key_time], keep=False).any():
        raise ValueError(f"DataFrame has duplicated ids and time points, please remove duplicates:\n"
                         f"{df[df[[key_id, key_time]].duplicated(keep=False)][[key_id, key_time]]}")
    df = df.copy()
    df[f'{feature}_1'] = df[feature].copy()
    df[f'{feature}_0'] = np.nan
    for uid_, df_ in df.groupby(key_id):
        df_ = df_.copy()
        df_ = df_.sort_values(key_time)
        ages = df_[key_time].values
        for i, age in enumerate(ages[:-shift]):
            # print(df_[df_[key_time] == ages[i + shift]].index, '###',
            #       df.loc[df_[df_[key_time] == ages[i + shift]].index, f'{feature}_0'],
            #       '###',  df_[key_time] == age )
            mask_age1 = (df_[key_time] == ages[i + shift]).values
            mask_age0 = (df_[key_time] == ages[i]).values
            # if mask_age1.sum() != mask_age0.sum():
            #     print("HMMM", mask_age1.sum(), mask_age0.sum())
            #
            # if mask_age1.sum() == 0:
            #     continue
            # print("ZZZZ", mask_age1, mask_age0,  df_.index[mask_age1], "####\n",
            #       df.loc[df_.index[mask_age1], f'{feature}_0'], "####\n",
            #       df_.loc[mask_age0, f'{feature}_1'].values)

            df.loc[df_.index[mask_age1], f'{feature}_0'] = df_.loc[mask_age0, f'{feature}_1'].values
    df[f'{feature}_delta'] = df[f'{feature}_1'] - df[f'{feature}_0']
    return df


def round_1digit_alt(x, max_dig=5):
    """
    Formats float < 1 to the format up to n-digits after float, if number smaller than 10^-N return exponential format

    Parameters
    ----------
    x : float
        float to format
    max_dig : int, optional
        number of digit to format in float notation

    Returns
    -------
        fmt : string
    """
    neg_log_x = -np.log10(x)
    if np.isfinite(neg_log_x):
        ndig = int(neg_log_x)
        if ndig < max_dig:
            fmt = "p=%%.%df" % (ndig + 1)
        else:
            fmt = "p=%.2e"
    else:
        fmt = "p=%.2e"
    return fmt % x


def calc_pvals_sign(data, x_shift=0, pval_func=None, alpha=0.05, scale_cap_leg=0.025):
    """ Draw caps and p-values with significance level `alpha` """
    # ax = plt.gca() if ax is None else ax

    if pval_func is None:
        pval_func = lambda x, y: scipy.stats.mannwhitneyu(x, y)[1]
    else:
        if not callable(pval_func):
            raise ValueError("pval_func should be callable")

    def pval_shift(level, max_height_diff, scale=0.1):
        """ Shifts caps by this value """
        return (level + 1) * max_height_diff * scale

    box_ind = np.arange(len(data), dtype=int)

    min_vals = np.array([np.nanpercentile(d,2) for d in data])
    max_vals = np.array([np.nanpercentile(d,98) for d in data])
    diff = max_vals.max() - min_vals.min()
    y_min_max = max_vals.max()  # the lowest position of pvalues cap
    cap_height = scale_cap_leg * diff

    comb = np.array(list(itertools.combinations(box_ind, 2)))
    diff_comb = np.diff(comb, axis=1)

    n_pvals = 0  # counter of significant pvalues
    pvals = list()
    for c in np.argsort(diff_comb.flatten()):
        i, j = comb[c]
        gr1 = data[i]
        gr2 = data[j]
        m1 = np.isfinite(gr1)
        m2 = np.isfinite(gr2)
        pval_ = pval_func(gr1[m1], gr2[m2])
        if pval_ < alpha:
            y = y_min_max + pval_shift(n_pvals, diff)
            pvals.append((round_1digit_alt(pval_), (i + x_shift, j + x_shift), y + cap_height, diff))
            n_pvals += 1
    return pvals
