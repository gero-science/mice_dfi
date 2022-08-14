__all__ = [
    'table_MPD_statistics',
    'table_MPD_survival',
]

import lifelines
import numpy as np
import pandas as pd
import scipy


def calc_cox_pval(df, age, sex='M', x='bio', mort_tte='mort_tte', mort_event='mort_event'):
    """ Returns concordance and pvalue """

    sex = sex if isinstance(sex, (list, tuple)) else [sex, ]
    if isinstance(age, (list, tuple)):
        assert len(age) == 2
        df = df[df.sex.isin(sex) & (df.age < age[1]) & (df.age > age[0])].copy()
    else:
        df = df[df.sex.isin(sex) & (df.age == age)].copy()
    df = df[[x, mort_tte, mort_event]].dropna()
    df[x] -= df[x].mean()
    df[x] /= df[x].std()

    if len(df) < 5:
        return np.nan, np.nan
    cph = lifelines.CoxPHFitter()
    cph = cph.fit(df, 'mort_tte', 'mort_event')
    return float(cph.summary['exp(coef)'].iloc[0]), tuple(np.exp(cph.confidence_intervals_.iloc[0].values)), float(cph.summary['p'].iloc[0])


def calc_pval(df, age, sex='M', x='bio', y='mort_tte', func=None):
    func = scipy.stats.pearsonr if func is None else func
    sex = sex if isinstance(sex, (list, tuple)) else [sex, ]
    if isinstance(age, (list, tuple)):
        assert len(age) == 2
        df = df[df.sex.isin(sex) & (df.age < age[1]) & (df.age > age[0])].copy()
    else:
        df = df[df.sex.isin(sex) & (df.age == age)].copy()
    df = df.dropna(subset=[x, y])
    return func(df[x].values, df[y].values)


def table_MPD_statistics(df, features, save_tex=False, save_xls=False):
    if isinstance(features, (pd.Series, np.ndarray)):
        features = features.tolist()
    elif isinstance(features, (list, tuple)):
        pass
    else:
        raise TypeError(f"Data type of features {type(features)} is incorrect")

    print("Used dataset and mice number")
    print(f"CBC features: {' '.join(features)}")

    df['# of animals'] = 1
    df['# of missing'] = df[features].isna().any(1).astype(int)

    peters4_starins = df[df.label == 'Peters4']['strain'].unique()

    df['PCA(age > 20 weeks)'] = (df.strain.isin(peters4_starins) & (df.age > 20) & ~(df[features].isna().any(1))).astype(int)
    df['PCA(age > 6 weeks)'] = (df.strain.isin(peters4_starins) & ~(df[features].isna().any(1))).astype(int)
    df['Training autoencoder'] = (~(df[features].isna().any(1))).astype(int)

    # Longitudinal slice
    uids = df.groupby('uid').size()[(df.groupby('uid').size() > 1)].index.values
    strains = df.groupby('strain_sex').size()[df.groupby('strain_sex').size() >= 8].index.values
    df['Training autoregression'] = (df.strain_sex.isin(strains) & df.uid.isin(uids) & ~(df[features].isna().any(1))).astype(int)

    stats_cols = ['# of animals', '# of missing', 'PCA(age > 20 weeks)', 'PCA(age > 6 weeks)',
                  'Training autoencoder', 'Training autoregression']
    group_cols = ['label', 'sex', 'age']
    stats_training = df[stats_cols + group_cols].groupby(group_cols).sum()
    stats_training.loc[('', '', "Total")] = stats_training.sum()

    if save_tex:
        stats_training.to_latex('train_missing.tex', multirow=True)
    if save_xls:
        with pd.ExcelWriter('train_missing.xlsx', engine="openpyxl") as writer:
            stats_training.to_excel(writer)
    return stats_training


def table_MPD_survival(df, biomarkers, cox=False, fpval=None, aggregate=False, fmt='pub',
                       alpha=0.05, return_latex=False):
    """ Tabel for lifespan """

    def table_format(x, fmt='pub'):
        if type(x) not in [int, float] and len(x) == 2:
            if fmt == 'pub':
                return '%.2f (<0.001)' % x[0] if x[1] < 0.001 else "%.2f (%.3f)" % (x[0], x[1])
            elif fmt == 'exp':
                return "%.2f (%.1e)" % (x[0], x[1])
        else:
            return str(x)
    def table_format_cox(x):
        if type(x) not in [int, float] and len(x) == 3:
            HR, CI, pval = x
            return "{:.2f}, ({:.2f}‒{:.2f}, p={:.1e})".format(HR, CI[0], CI[1], pval)
        else:
            return str(x)

    def table_format_latex(x, fmt='pub'):
        if type(x) not in [int, float] and len(x) == 2:
            if fmt == 'pub':
                res = '%.2f (<0.001)' % x[0] if x[1] < 0.001 else "%.2f (%.3f)" % (x[0], x[1])
            elif fmt == 'exp':
                res = "%.2f (%.1e)" % (x[0], x[1])
            else:
                raise ValueError(f"{fmt=}")
            if x[1] < alpha:
                res = '\\textbf{%s}' % res
            return res
        else:
            return str(x)

    def color_cell(x):
        x = x.split()
        if len(x) > 1:
            x = x[1].strip('(').strip(')').strip('<')
            if float(x) < 0.05:
                return 'font-weight: bold'
            else:
                return 'font-weight: '
        else:
            return 'font-weight: '

    def color_cell_cox(x):
        x = x.split()
        if len(x) > 1:
            x = x[2].strip('p=').strip(')').strip('<')
            if float(x) < 0.05:
                return 'font-weight: bold'
            else:
                return 'font-weight: '
        else:
            return 'font-weight: '

    if return_latex and cox:
        raise NotImplementedError("`return_latex=True` option is immutable with `cox=True` ")

    if fpval is None:
        fpval = scipy.stats.spearmanr
    ages = [26, 52, 78, ]
    sex_list = ['M', 'F']

    index_label = ['Cohort size 1', ]
    index_label += ['{:s} (all mice)'.format(biomarker) for biomarker in biomarkers]

    index_label += ['Cohort size 2']
    index_label += ['{:s} (IGF1 subset)'.format(biomarker) for biomarker in biomarkers]
    index_label += ['IGF1', 'Body weight']

    res = pd.DataFrame(index=index_label, dtype='O')
    for age_ in ages:
        for sex_ in sex_list:
            res['%s (%d w)' % (sex_, age_)] = ((None, None),) * len(index_label)
    df_ = df[(df.age >= 26)]
    if aggregate:
        df_ = df_.groupby(['age', 'sex', 'strain']).median().reset_index()
        df_['mort_tte'] = df_['lifespan']
        df_['mort_event'] = 1.

    m = None
    for s_ in sex_list:
        m_ = ((df_[(df_.sex.isin(list(s_)))]['lifespan'] > 0) & (df_.sex.isin(list(s_)))).values
        m = m_ if m is None else m | m_
    df_ = df_[m]

    print("N strains males %d, N strains females %d" % \
          (len(df_[df_.sex == 'M'].strain.unique()), len(df_[df_.sex == 'F'].strain.unique())))
    # df_ = df_[df_.bw<60]
    merged_df = df_[(~df_.igf1.isna()) | (~df_.igf1.isna())]

    for age_ in ages:
        for sex_ in sex_list:
            sex_age_ = '%s (%d w)' % (sex_, age_)
            res.loc['Cohort size 1', sex_age_] = len(df_[(df_.sex.isin(list(sex_))) &
                                                         (df_.age == age_) &
                                                         ~df_[biomarkers].isna().all(axis=1) &
                                                         (df_.mort_event == 1)
                                                         ])
            for bio_name in biomarkers:
                if cox:
                    pvals = calc_cox_pval(df_, age_, sex_, x=bio_name)
                else:
                    pvals = calc_pval(df_[df_.mort_event == 1], age_, sex_, x=bio_name, func=fpval)
                res.at['%s (all mice)' % bio_name, sex_age_] = pvals

            res.loc['Cohort size 2', sex_age_] = len(merged_df[(merged_df.sex.isin(list(sex_))) &
                                                               (merged_df.age == age_) &
                                                               ~merged_df[biomarkers].isna().all(axis=1) &
                                                               (merged_df.mort_event == 1)
                                                               ])
            for bio_name in biomarkers:
                if cox:
                    pvals = calc_cox_pval(merged_df, age_, sex_, x=bio_name)
                else:
                    pvals = calc_pval(merged_df[merged_df.mort_event == 1], age_,
                                      sex_, x=bio_name, func=fpval)
                res.at['%s (IGF1 subset)' % bio_name, sex_age_] = pvals
            if cox:
                pvals = calc_cox_pval(merged_df, age_, sex_, x='igf1')
            else:
                pvals = calc_pval(merged_df[merged_df.mort_event == 1], age_,
                                  sex_, x='igf1', func=fpval)
            res.at['IGF1', sex_age_] = pvals
            if cox:
                pvals = calc_cox_pval(merged_df, age_, sex_, x='bw')
            else:
                pvals = calc_pval(merged_df[merged_df.mort_event == 1], age_,
                                  sex_, x='bw', func=fpval)
            res.at['Body weight', sex_age_] = pvals

    if cox:
        df_repr = res.applymap(lambda x: table_format_cox(x)).style.applymap(color_cell_cox)
    else:
        df_repr = res.applymap(lambda x: table_format(x, fmt)).style.applymap(color_cell)

    if return_latex and not cox:
        return df_repr, res.to_latex(formatters=[lambda x: table_format_latex(x, fmt)] * len(res.columns),
                                     bold_rows=True,
                                     escape=False)
    else:
        return df_repr
