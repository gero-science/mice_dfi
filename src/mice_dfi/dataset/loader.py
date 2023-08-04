# -*- coding: utf8 -*-
"""
All loaders for mice datasets.
"""
__all__ = [
    'load_meta_blood',
    'load_full_meta',
    'sanity_check',
    'filter_CBC_dataset',
    'load_lifespan_itp',
    'fill_wbc_counts'
]

import os
import glob
import pandas as pd
import numpy as np

_GLOBAl_DIR_PATH = os.path.dirname(__file__) + '/raw'
_GLOBAl_DATA_PATH = dict(
    CBC='CBC',
    bone='bone',
    serum='serum',
    gait='gait',
    lifespan='lifespan',
)

_map_cbc_names = {
    'id': 'animal id',
    'mpv': 'mpv (fl)',
    'mhgb': 'hb (g/dl)',
    'hgb': 'hb (g/dl)',
    'pctretic': 'retic %',
    'pct_retic': 'retic %',
    'plt': 'plt (k/ul)',
    'wbc': 'wbc (k/ul)',
    'rbc': 'rbc (m/ul)',
    'rdw': 'rdw %',
    'mchc': 'mchc (g/dl)',
    'mcv': 'mcv(fl)',
    'mch': 'mch (pg)',
    'hct': 'hct %',
    'pcthct': 'hct %',
    'eos': 'eo (k/ul)',
    'pcteos': 'eo %',
    'pct_eos': 'eo %',
    'pcte': 'eo %',
    'neut': 'ne (k/ul)',
    'pct_neut': 'ne %',
    'pctn': 'ne %',
    'pctneut': 'ne %',
    'lym': 'ly (k/ul)',
    'lymp': 'ly (k/ul)',
    'lymph': 'ly (k/ul)',
    'pctlymp': 'ly %',
    'pctlym': 'ly %',
    'pctl': 'ly %',
    'pctlymph': 'ly %',
    'pct_lymp': 'ly %',
    'pct_lym': 'ly %',
    'pct_lymph': 'ly %',
    'mono': 'mo (k/ul)',
    'pctmono': 'mo %',
    'pct_mono': 'mo %',
    'pctm': 'mo %',
    'retic': 'retic (m/ul)',
    'gra': 'gr (k/ul)',
    'pct_gra': 'gr %',
}

_fix_peters = {
    # Peters1/2 datasets have wrong naming for WBC (use percents instead of absolute values)
    'neut': 'pct_neut',
    'lym': 'pct_lym',
    'mono': 'pct_mono',
    'eos': 'pct_eos',
}


def set_global_dir(path):
    _GLOBAl_DIR_PATH = path


def set_global_data(**kwargs):
    for key, val in kwargs.items():
        _GLOBAl_DATA_PATH[key] = val


def get_global_dir():
    return _GLOBAl_DIR_PATH


def preprocess_mouse_data(df, is_longitudinal=False, strain='SWR/J', fix_tte=False, fill_defaults=True,
                          ignore_keys=None, append_keys=None):
    def get_float_X(X):
        X = np.array(X, dtype=np.str_)
        for i in range(X.shape[0]):
            # for j in range(X.shape[1]):
            try:
                X[i] = np.float64(X[i])
            except ValueError:
                X[i] = np.nan
        X = np.array(X, dtype=np.float64)
        X[X > 1e10] = np.nan
        X[X < -1] = np.nan
        X[X == 999999.] = np.nan
        # print 'Converted X: ', X.shape, '\n'
        return X

    _ignore_fields = ['box', 'age_mo', 'week', 'time']
    if ignore_keys is None:
        pass
    elif isinstance(ignore_keys, (list, tuple)):
        _ignore_fields.extend(list(ignore_keys))
    else:
        raise TypeError(f"{type(ignore_keys)=}")

    _fields_tuples = [('uid', 'animal id'), ('uid', 'uid'), ('id', 'mouse n'), ('group', 'group'),
                      ('sex', 'sex'), ('DOB', 'dob'), ('DOD', 'dod'), ('age', 'age'),
                      ('mort_age', 'age at death'), ('strain', 'strain'),
                      ('study_group', 'study group'), ('label', 'label'), ('group', 'treatment')]
    _int_types = []
    _float_type = ['age', 'age at death']
    _string_types = ['strain', 'sex', 'group', 'treatment', 'label', 'animal id',
                     'mouse n', 'study group', 'uid']
    _date_types = ['dob', 'dod']
    _default_values = {'group': 'Control', 'DOB': -1, 'DOD': -1, 'mort_age': np.NaN, 'strain': strain}

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    nsamples = len(df)

    # Required fields
    data_dict = dict()
    used_fields = []
    for key, val in _fields_tuples:
        if val in df.columns:
            values = df[val].values
            if val in _int_types:
                values = np.array(values, dtype=int)
            elif val in _float_type:  # float types
                values = np.array(values, dtype=np.float32)
            elif val in _string_types:  # String types
                values = np.array(values, dtype=str)  #
                values = np.char.array(values)
            elif val in _date_types:  # String types
                values = pd.to_datetime(values)
            else:
                raise ValueError("Type for pair key=%s, val=%s is not set" % (key, val))
            used_fields += [val]
            if key in ['uid', 'id']:
                values = values.lower().strip()
            data_dict[key] = values
        else:
            continue

    # Sanity checks
    if not (data_dict['age'] > 0).all():
        raise ValueError('Age should be positive')
    else:
        data_dict['age'] = data_dict['age'].astype(int)

    # Set defaults
    if 'uid' not in data_dict:
        data_dict['uid'] = data_dict['id']

    if append_keys is not None:
        for key in append_keys:
            data_dict[key] = df[key].values
    if fill_defaults:
        for key, val in _default_values.items():
            if key not in data_dict:
                data_dict[key] = np.array([val] * nsamples, dtype=type(val))

    # Set features
    genes = df.columns.difference(used_fields + _ignore_fields + list(append_keys)).values
    res = pd.DataFrame(data=data_dict, index=df.index)
    res = pd.concat((res, df[genes].apply(get_float_X, raw=True, axis=1)), axis=1)

    # Replace tiny animal IDs to avid collision
    tiny_uid = (res.uid.str.len() <= 4)
    res.loc[tiny_uid, 'uid'] = res[tiny_uid]['label'] + '_' + res[tiny_uid]['strain'] + '_' + res[tiny_uid]['uid']

    # Set multiindex
    ind_cols = ['uid', 'age']
    res = res.set_index(ind_cols)

    genes = np.setdiff1d(genes, ['pfi'])
    if is_longitudinal is True:
        unique_ages = np.unique(res.reset_index()[ind_cols[1]])
        # Npoint = len(unique_ages)
        uniq_samples = np.unique(res.reset_index()[ind_cols[0]])
        mlt_ind = pd.MultiIndex.from_product(
            [list(uniq_samples), list(unique_ages)], names=ind_cols)
        res = res.reindex(mlt_ind)
        res = res.sort_index(level=1)

        # add missing info
        columns = list(data_dict.keys())
        for i in ind_cols:
            columns.remove(i)

        for i in res[res[columns].isna().all(axis=1)].index:
            res.loc[i, columns] = res.loc[(i[0], mlt_ind[0][1]), columns]

    if 'mort_age' in res.columns:
        res['mort_tte'] = res.reset_index()['mort_age'].values - res.reset_index()['age'].values
        res['mort_event'] = res['mort_tte'].apply(lambda x: x / x)

    if fix_tte:
        # Use longitudinal data and fill consequent points with mort_tte and mort_event
        res = res.reset_index()
        uid_long = res.uid.value_counts()[res.uid.value_counts() > 1].index
        lst = ['mort_age', 'mort_tte', 'mort_event']
        for uid_, df_ in res[res.uid.isin(uid_long)].groupby('uid'):
            if df_.mort_event.isna().all():
                df_ = df_.sort_values('age')
                age_surv_max = df_['age'].max()
                mask = df_.age < age_surv_max
                df_.loc[mask, 'mort_age'] = age_surv_max
                df_.loc[mask, 'mort_event'] = 0.
                df_.loc[mask, 'mort_tte'] = df_.loc[mask, 'mort_age'] - df_.loc[mask, 'age']
                for ii in lst:
                    res.loc[df_.index, ii] = df_[ii].values
            elif df_.mort_event.isna().any():
                raise ValueError("Something went wrong", df_)
            else:
                pass
        res = res.set_index(ind_cols)

    return res, genes


def restore_full_column_wbc(df):
    _wbc_names = ['ly', 'ne', 'gr', 'mo', 'ba', 'eo']
    _wbc_unit = '(k/ul)'
    wbc_key = 'wbc %s' % _wbc_unit
    if wbc_key not in df.columns:
        raise RuntimeError("No WBC data, is it blood dataset?")
    for name in _wbc_names:
        counts_name = '%s %s' % (name, _wbc_unit)
        pct_name = '%s %%' % name

        if counts_name in df.columns and pct_name not in df.columns:
            df[pct_name] = 100. * df[counts_name] / df[wbc_key]
        elif counts_name not in df.columns and pct_name in df.columns:
            df[counts_name] = 0.01 * df[pct_name] * df[wbc_key]
        else:
            pass
    return df


def fill_wbc_counts(df):
    wbc_fields = {'eo (k/ul)': 'eo %', 'ly (k/ul)': 'ly %', 'mo (k/ul)': 'mo %', 'ne (k/ul)': 'ne %'}
    rbc_fields = {'retic (m/ul)': 'retic %'}

    def impute_zeros(df_, key):
        series = df_[key].values.copy()
        mnorm = series > 0
        if mnorm.sum() == 0:
            return
        mzeros = series == 0
        series[mzeros] = series[mnorm].min()
        df_[key] = series

    for key, val in wbc_fields.items():
        if key in df.columns:
            impute_zeros(df, key)
        if val in df.columns:
            impute_zeros(df, val)

    df = restore_full_column_wbc(df)

    df['gr %'] = 100. - df['ly %'] - df['mo %']
    mask = df['wbc (k/ul)'].isna().values
    for col_to_fill in wbc_fields.keys():
        if col_to_fill not in df.columns.values.tolist():
            continue
        m = df[col_to_fill].isna().values * ~mask
        df.loc[m, col_to_fill] = df.loc[m, wbc_fields[col_to_fill]] * df.loc[m, 'wbc (k/ul)'] / 100.
    mask = df['rbc (m/ul)'].isna().values
    for col_to_fill in rbc_fields.keys():
        if col_to_fill not in df.columns.values.tolist():
            continue
        m = df[col_to_fill].isna().values * ~mask
        df.loc[m, col_to_fill] = df.loc[m, rbc_fields[col_to_fill]] * df.loc[m, 'rbc (m/ul)'] / 100.

    df['gr (k/ul)'] = df['wbc (k/ul)'] - df['ly (k/ul)'] - df['mo (k/ul)']
    return df


def sanity_check(df, genes=None, abs_tol_wbc=0.4, rel_tol_wbc=0.1, tol_rbc=0.05, verbose=0):
    missed_pass = True
    values_pass = True
    if genes is not None:
        nsamples = df.shape[0]
        genes_missed = nsamples - df[genes].count()
        if genes_missed.values[genes_missed > 0].any():
            if verbose == 2:
                print('Found missing values')
                print(genes_missed[genes_missed > 0])
                genes_ = genes_missed[(genes_missed > 0) & (genes_missed <= 10)].index
                for g_ in genes_:
                    print(df[df[g_].isna()][['label', 'age', 'uid', g_]])
            missed_pass = False

    def calc_from_pct(x, pct):
        return x * pct / 100.

    def check_wbc_pct(wbc, x, xpct):
        """ Checks if values calculated from total WBC and cell percentage match absolute value """
        m = np.isfinite(wbc) & np.isfinite(x) & np.isfinite(xpct)
        x_pred = calc_from_pct(wbc[m], xpct[m])
        diff = (np.abs(x[m] - x_pred) / x[m] > rel_tol_wbc) & (np.abs(x[m] - x_pred) > abs_tol_wbc)
        index = np.zeros_like(m, dtype=bool)
        index[m] |= diff
        return index

    def calc_mcv(hct, rbc):
        return hct / rbc * 10.

    def calc_mch(hb, rbc):
        return hb / rbc * 10.

    def calc_mchc(hb, hct):
        return hb / hct * 100.

    def check_rbc(hb, rbc, hct, mcv, mch, mchc, tol=tol_rbc):
        m = np.isfinite(hb) * np.isfinite(rbc) * np.isfinite(hct)
        mcv_pred = calc_mcv(hct[m], rbc[m])
        diff_mcv = np.abs(mcv[m] - mcv_pred) / mcv[m] > tol

        mch_pred = calc_mch(hb[m], rbc[m])
        diff_mch = np.abs(mch[m] - mch_pred) / mch[m] > tol

        mchc_pred = calc_mchc(hb[m], hct[m])
        diff_mchc = np.abs(mchc[m] - mchc_pred) / mchc[m] > tol

        index = np.zeros_like(m, dtype=bool)
        index[m] |= diff_mcv
        index[m] |= diff_mch
        index[m] |= diff_mchc

        return index, dict(mcv_pred=mcv_pred[index[m]], mch_pred=mch_pred[index[m]], mchc_pred=mchc_pred[index[m]])

    _wbc_names = ['ly', 'ne', 'gr', 'mo']
    _wbc_unit = '(k/ul)'
    ind_wbc_pct_problem = np.zeros(len(df), dtype=bool)
    for name in _wbc_names:
        wbcn = 'wbc %s' % _wbc_unit
        datan = '%s %s' % (name, _wbc_unit)
        pctn = '%s %%' % name
        if datan not in df.columns:
            continue
        ind_wbc_pct_problem = check_wbc_pct(df[wbcn].values, df[datan].values, df[pctn].values)
        if sum(ind_wbc_pct_problem) > 0:
            values_pass = False
            if verbose:
                print(f"Found {sum(ind_wbc_pct_problem)} samples with WBC data inconsistency")
            if verbose == 2:
                df_ = df[ind_wbc_pct_problem].copy()
                df_['%s expected' % name] = calc_from_pct(df_[wbcn].values, df_[pctn].values)
                print('Found data inconsistency for %s' % name)
                print(df_[['label', 'age', 'uid', wbcn, pctn, datan, '%s expected' % name]])

    # Checks that sum of WBC components match total WBC
    # The WBC descriptors might use granulocytes or NE+EO formula
    descr_gr = ['ly (k/ul)', 'mo (k/ul)', 'gr (k/ul)']
    descr_ne = ['ly (k/ul)', 'mo (k/ul)', 'ne (k/ul)']
    if 'eo (k/ul)' in df.columns:
        descr_ne.append('eo (k/ul)')

    ind_wbc_tot_problem = np.zeros(len(df), dtype=bool)
    for d in [descr_gr, descr_ne]:
        try:
            df[d]
        except KeyError:
            continue
        wbc_tot_diff = df['wbc (k/ul)'].values - np.nansum(df[d].values, axis=1)  # .sum(axis=1).values
        ind_wbc_tot_problem = np.abs(wbc_tot_diff) / df['wbc (k/ul)'].values > rel_tol_wbc
        if sum(ind_wbc_tot_problem) > 0:
            values_pass = False
            if verbose:
                print(f"Found {sum(ind_wbc_tot_problem)} samples with WBC total number not equal sum of components")
            if verbose == 2:
                df_ = df[ind_wbc_tot_problem].copy()
                df_['wbc expected'] = df_[d].sum(axis=1).values
                print('Found data inconsistency for wbc total')
                print(df_[['label', 'age', 'uid', 'wbc (k/ul)', 'wbc expected']])

    descr_gr = ['ly %', 'mo %', 'gr %']
    descr_ne = ['ly %', 'mo %', 'ne %']
    if 'eo %' in df.columns:
        descr_ne.append('eo %')
    for d in [descr_gr, descr_ne]:
        try:
            df[d]
        except KeyError:
            continue
        wbc_tot_diff = 100. - df[d].values.sum(axis=1)  # sum(axis=1).values
        ind_ = np.abs(wbc_tot_diff) / 100 > rel_tol_wbc
        if sum(ind_) > 0:
            values_pass = False
            if verbose:
                df_ = df[ind_].copy()
                df_['wbc % expected'] = df_[d].sum(axis=1).values
                print('Found data inconsistency for wbc pct')
                print(df_[['label', 'age', 'uid', 'wbc % expected']])

    ind_rbc_problem, calc_ = check_rbc(df['hb (g/dl)'].values, df['rbc (m/ul)'].values, df['hct %'].values,
                                       df['mcv(fl)'].values, df['mch (pg)'].values, df['mchc (g/dl)'].values)

    if sum(ind_rbc_problem) > 0:
        values_pass = False
        if verbose:
            print(f'Found data inconsistency for red blood in {sum(ind_rbc_problem)} samples')
        if verbose == 2:
            df_ = df[ind_rbc_problem].copy()
            for key, val in calc_.items():
                df_[key] = val
            print(df_[['label', 'age', 'uid', 'mcv(fl)', 'mcv_pred',
                       'mch (pg)', 'mch_pred', 'mchc (g/dl)', 'mchc_pred']])
    ind_problem = np.logical_or(ind_wbc_pct_problem, ind_wbc_tot_problem, ind_rbc_problem)

    return values_pass and missed_pass, ind_problem


def filter_CBC_dataset(df, subset, drop_fraction=0.1, hb_thresh=5.):
    """ Apply `sanity_check` to each dataset and removed datasets with many corrupted samples """
    drop_dataset = []
    for gr_, df_ in df.groupby('label'):
        _, ind_ = sanity_check(df_, genes=subset, rel_tol_wbc=0.1, abs_tol_wbc=0.2, verbose=0)
        fraction_bad = sum(ind_) / len(df_)
        if fraction_bad > drop_fraction:
            drop_dataset.append(gr_)
        else:
            print(gr_, 'OK')

    for ds in drop_dataset:
        print(f"Excluding dataset {ds}, failed sanity check")
        df = df[df.label != ds].copy()

    outliers = (df['hb (g/dl)'] < hb_thresh).values
    df = df[~outliers].copy()
    print("Load DataFrame, shape:", df.shape, "outliers removed:", outliers.sum())

    return df


def load_lifespan_yuan(fname=None):
    fname = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['lifespan'], 'Yuan2_strainmeans.csv') \
        if fname is None else fname
    meta_df_lifespan = pd.read_csv(fname)
    meta_df_lifespan = meta_df_lifespan[meta_df_lifespan.varname == 'lifespan_median'].copy()
    meta_df_lifespan['sex'] = meta_df_lifespan['sex'].apply(
        lambda x: x.upper())
    meta_df_lifespan['strain'] = meta_df_lifespan['strain'].apply(
        lambda x: x.lower())

    meta_df_lifespan.rename(columns={'mean': 'lifespan'}, inplace=True)
    meta_df_lifespan['lifespan'] = meta_df_lifespan['lifespan'] / 7.
    meta_df_lifespan = meta_df_lifespan.drop_duplicates()
    return meta_df_lifespan


def load_meta_blood(path=None):
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['CBC']) if path is None else path
    meta_lst = []
    for f_ in glob.glob(os.path.join(path, '*.csv')):
        dts_name = os.path.basename(f_).split('.')[0]
        # if dts_name in ['HMDPpheno5']:
        #     continue
        meta_df = pd.read_csv(f_)
        meta_df['label'] = dts_name
        meta_df.columns = meta_df.columns.str.lower()
        if dts_name == 'Peters1':
            meta_df = meta_df.rename(columns=_fix_peters)
            meta_df['age'] = 11.
        elif dts_name == 'Peters2':
            meta_df = meta_df.rename(columns=_fix_peters)
            meta_df['age'] = 11.
        elif dts_name == 'CGDpheno1':
            meta_df['age'] = 11.
        elif dts_name == 'Svenson3':
            meta_df['age'] = 7.
        elif dts_name == 'CGDpheno3':
            meta_df['age'] = 7.
        elif dts_name == 'Justice2':
            meta_df['age'] = 14.
        elif dts_name == 'Jaxpheno4':
            meta_df, _ = read_long_mice_jax(meta_df, [8, 16], fmt='%d', age_scale=1., key_id='id')
            meta_df = meta_df.dropna(thresh=int(meta_df.shape[1] * 0.6))
            meta_df = meta_df.rename(columns={'uid': "animal id"})
        elif dts_name == 'Petkova1':
            meta_df['age'] = meta_df['age'].apply(
                lambda x: np.round(x * 30.4 / 7))
            pass
        elif dts_name == 'HMDPpheno5':
            meta_df['age'] = 16.
        elif dts_name == 'Lake1':
            meta_df['age'] = 7.
            meta_df = meta_df.dropna(thresh=int(meta_df.shape[1] * 0.4))
        elif dts_name == 'Peters4':
            meta_df['age'] = meta_df['age'].apply(
                lambda x: np.round(x * 30.4 / 7))
        else:
            raise ValueError('Unknown dataset %s' % dts_name)
        meta_df = meta_df.rename(columns=_map_cbc_names)
        meta_df['sex'] = meta_df['sex'].apply(lambda x: x.upper())
        meta_lst.append(meta_df)

    if len(meta_lst) == 0:
        raise ValueError('No datasets found in %s' % path)
    meta_df = pd.concat(meta_lst, ignore_index=True)
    # Select the most populated blood params
    notnans = meta_df.count()
    notnans = notnans[notnans > len(meta_df) * 0.55]
    genes = notnans.index
    genes = genes.difference(['sex', 'age', 'label', 'animal id', 'strain']).tolist()
    return meta_df, genes


def load_full_meta(debug=True, signal='blood', path=None):
    """ Load MetaMice returns pandas DataFrame
    Parameters:
        signal : str, `serum`, `biochem`, `bone`
    """

    fix_tte = False
    if signal == 'blood':
        print("Loading CBC...")
        meta_df, genes_g = load_meta_blood(path)
        fix_tte = True
    elif signal == 'serum':
        print("Loading serum...")
        meta_df, genes_g = load_dataset_igf(path)
        genes_g = genes_g.astype(str)
        genes_g = np.char.asarray(genes_g).lower()
    elif signal == 'bone':
        print("Loading bone...")
        meta_df, genes_g = load_dataset_bone(path)
        genes_g = genes_g.astype(str)
        genes_g = np.char.asarray(genes_g).lower()
    elif signal == 'gait':
        print("Loading gait...")
        meta_df, genes_g = load_dataset_gait(path)
        genes_g = genes_g.astype(str)
        genes_g = np.char.asarray(genes_g).lower()
    elif signal == 'biochem':
        print("Loading biochemistry...")
        meta_df, genes_g = load_dataset_biochem(path)
        genes_g = genes_g.astype(str)
        genes_g = np.char.asarray(genes_g).lower()
    else:
        raise ValueError('Desriptor %s is not supported. Availbale options: `serum`, `biochem`, `bone`')

    #  Load mice death if known
    print("Appending lifespan...")
    f_death_csv = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['lifespan'], 'Yuan2.csv') \
        if path is None else path
    all_death = pd.read_csv(f_death_csv)
    # all_death = all_death.rename(columns={'animal_id': 'animal id'})
    all_death = all_death.rename(columns=_map_cbc_names)
    all_death['animal id'] = all_death['animal id'].astype(str).str.lower()
    all_death['age at death'] = all_death['lifespandays'] / 7.  # Scale to weeks
    all_death['label'] = 'Peters4'
    meta_df['animal id'] = meta_df['animal id'].astype(str).str.lower()

    meta_df = meta_df.merge(all_death[['animal id', 'label', 'age at death']], 'left', on=['animal id', 'label'])
    del all_death

    label = meta_df['label'].values
    meta_df, meta_g = preprocess_mouse_data(meta_df, fix_tte=fix_tte)
    meta_df['label'] = label
    meta_df['strain'] = meta_df['strain'].apply(lambda x: x.lower())
    meta_df['strain_sex'] = meta_df['strain'] + '_' + meta_df['sex']

    # Life span
    print("Appending strains mean lifespan...")
    meta_df_lifespan = load_lifespan_yuan()
    meta_df_full = pd.merge(meta_df.reset_index(), meta_df_lifespan[['sex', 'strain', 'lifespan']], on=[
        'sex', 'strain'], how='left')  # WARNING: inner is used

    # Impute missing values
    if signal == 'blood':
        meta_df = fill_wbc_counts(meta_df)
        meta_df_full = fill_wbc_counts(meta_df_full)
        genes_g = np.intersect1d(meta_g, genes_g)
        genes_g = np.concatenate((genes_g, ['gr %', 'gr (k/ul)']))

        # Clean unused features
        genes_g = np.setdiff1d(genes_g, ['baso', 'luc', 'chr', 'pctbaso', 'pctluc', 'mpv (fl)'])

        # This is aggregated as 'gr %', 'gr (k/ul)', we do not need them
        genes_g = np.setdiff1d(genes_g,
                               ['eo %', 'eo (k/ul)', 'ne (k/ul)', 'ne %',
                                'retic %', 'retic (m/ul)', 'mo (k/ul)', 'mo %'])

        # drop everything if <50% filled values
        meta_df_full = meta_df_full.dropna(subset=genes_g, thresh=int(len(genes_g) * 0.5))
        if debug:
            pd.options.display.max_rows = 500
            sanity_check(meta_df_full, genes=genes_g, verbose=2, rel_tol_wbc=0.1, abs_tol_wbc=0.2)

    if debug:
        # No old mice is available for strains without life span
        diff_strain = np.setdiff1d(meta_df.reset_index().strain.astype(str), meta_df_full.strain.astype(str))
        print("Not included strains")
        print(meta_df[meta_df.strain.isin(diff_strain)].dropna(subset=genes_g).
              groupby('strain').size().sort_values(ascending=False))
    return meta_df_full, genes_g


def read_long_mice_jax(fname, ages, fmt='_M%02d', age_scale=1., key_id='id'):
    """ Reads specific phenome.jax format. Xage1 Xage2 Xage3..."""

    if isinstance(fname, str):
        df = pd.read_csv(fname)  # there is duplicate for mouse id '00068cf1e2', it was mannualy fixed for male mouse
    elif isinstance(fname, (pd.DataFrame)):
        df = fname.copy()
    else:
        raise TypeError('Patrameter fname has wrong type %s' % (type(fname)))
    assert key_id in df.columns, f"Index {id} not found in df columns {df.columns}"
    if 'strain' in df.columns:
        df['strain'] = df['strain'].str.lower()
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.upper()
    if 'Animal ID' in df.columns:
        df = df.rename(columns={'Animal ID': key_id})
    df['age'] = ages[0]
    df = df.replace('WD', np.NaN)
    df = df.drop_duplicates([key_id, 'age'])
    df = df.set_index([key_id, 'age'])
    df = df.reindex(
        pd.MultiIndex.from_product([df.reset_index()[key_id].unique(), ages], names=[key_id, 'age'], sortorder=0))

    # find teplates in data
    template = [fmt % (i,) for i in ages]
    labels = []
    labels_to_drop = []

    for c_ in df.columns:
        for i, t_ in enumerate(template):
            pos = c_.rfind(t_)
            if pos > 0 and pos >= len(c_) - len(t_):
                new_name = c_[:pos]
                if new_name not in labels:
                    df[new_name] = np.NaN
                    labels.append(new_name)
                df.loc[pd.IndexSlice[:, ages[i]], new_name] = df.loc[pd.IndexSlice[:, ages[0]], c_].values
                labels_to_drop.append(c_)

    labels_to_keep = df.columns.difference(labels_to_drop + labels)
    df = df.drop(labels=labels_to_drop, axis=1)  # drop columns repeated columns
    df = df.reset_index()
    for key in labels_to_keep:
        if key not in df.columns:
            continue
        df[key] = df.loc[:, key].fillna(method='ffill')  # Fill gaps for strain and sex

    df['age'] = (df['age'].values * age_scale).astype(int)
    df = df.rename(columns={key_id: 'uid'})
    df = df.dropna(axis=1, thresh=int(0.05 * df.shape[0]))  # drop sparce columns (<5% of dataset)
    labels = df.columns.intersection(labels).values
    # df[labels] = df[labels].astype(float)
    return df, labels


# Datasets
def load_dataset_igf(path=None):
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['serum']) if path is None else path
    df, genes = read_long_mice_jax(os.path.join(path, 'Yuan1.csv'), [6, 12, 18], age_scale=30.5 / 7)
    df['IGF1_log'] = df['IGF1'].apply(np.log)
    df['label'] = 'Yuan1'
    df['animal id'] = df['uid'].values
    df['mouse n'] = df['uid'].values

    return df, genes


def load_dataset_bone(path=None):
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['bone']) if path is None else path
    df, genes = read_long_mice_jax(os.path.join(path, 'Ackert1.csv'), [6, 12, 20], age_scale=30.5 / 7)
    df['label'] = 'Ackert1'
    df['animal id'] = df['uid'].values
    df['mouse n'] = df['uid'].values
    return df, genes


def load_dataset_gait(path=None):
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['gait']) if path is None else path
    df, genes = read_long_mice_jax(os.path.join(path, 'Seburn2.csv'), [6, 12, 18, 24], age_scale=30.5 / 7)
    df['label'] = 'Seburn2'
    df['animal id'] = df['uid'].values
    df['mouse n'] = df['uid'].values
    return df, genes


def load_dataset_biochem(path=None):
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['serum']) if path is None else path
    df, genes = read_long_mice_jax(os.path.join(path, 'Yuan3.csv'), [6, 12, 18], age_scale=30.5 / 7)
    df['label'] = 'Yuan3'
    df['animal id'] = df['uid'].values
    df['mouse n'] = df['uid'].values
    return df, genes


def load_lifespan_itp(path=None):
    """ Data from ITP1: Interventions Testing Program """
    path = os.path.join(_GLOBAl_DIR_PATH, _GLOBAl_DATA_PATH['lifespan']) if path is None else path
    all_files = glob.glob(f'{path}/itp1/Lifespan*.xlsx')
    if len(all_files) == 0:
        raise IOError(f"No fileles found in {path}/itp1")
    else:
        print(f'Total files: {len(all_files)}')

    lst = []
    for fname in all_files:
        data = pd.read_excel(fname)
        data = data.rename(columns={'Dead': 'dead', 'Age': 'age', 'age(days)': 'age',
                                    'Status': 'status'})
        lst.append(data)

    data = pd.concat((lst), ignore_index=True)
    print("Total # of animals:", data.shape[0])
    data = data[data.group == 'Control'].copy()
    print("# of controls:", data.shape[0])
    data = data[data.dead == 1].copy()
    print("# of controls with known lifespan:", data.shape[0])
    data['age'] = data['age'] / 7
    return data


def load_glob_by_mask(path, date_ref, prefix, skiprows=0, age_in_file=False):
    day0 = datetime.datetime.strptime(date_ref, '%Y_%m_%d')
    df_lst = []
    file_lst = glob.glob(path + '/%s_*_preproc.xls*' % prefix)
    file_lst = np.sort(file_lst, )
    for i, name in enumerate(file_lst):
        pattern = re.findall('%s_([0-_]*)_preproc' % prefix, name)[0]
        ndays = (datetime.datetime.strptime(pattern, '%Y_%m_%d') - day0).days
        data = pd.read_excel(name, header=skiprows)
        data.columns = data.columns.str.lower()
        data['uid'] = data['uid'].astype(str)
        data['%s_num' % prefix] = i
        if age_in_file:
            data['age'] += int(ndays / 7.)
        else:
            data['age'] = int(ndays / 7.)
        df_lst.append(data.copy())

    data = pd.concat(df_lst, ignore_index=True)
    data, genes = preprocess_mouse_data(data, is_longitudinal=False, strain='C57BL/6J', fill_defaults=False)
    genes = np.setdiff1d(genes, ['%s_num' % str.lower(prefix)])
    data = data.reset_index()
    return data, genes

