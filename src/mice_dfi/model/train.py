import argparse
import os
import tensorflow as tf
import numpy as np
from ruamel import yaml
import warnings
import mice_dfi.dataset as mdata
import mice_dfi.model as mmodel
import mice_dfi.plots.utils as mutils

warnings.filterwarnings('ignore')
np.set_printoptions(precision=5, suppress=True)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory path where runs are stored")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=800, help="Number of epochs to train the model")
    parser.add_argument("--batch-size", type=int, default=15, help="Number of samples per gradient update")
    parser.add_argument("--tag", type=str, default='run', help="Name for the run, is used for logs output directory")
    parser.add_argument("--save-every", type=int, default=20, help="Saves model dumps these every steps")
    parser.add_argument("--tb", "--tensorboard", action='store_true', help="Add log tracking in tensorboard")
    return parser


if __name__ == '__main__':
    args = _get_parser().parse_args()

    savedir = mmodel.make_rec_dir(args.output)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load CBC features
    meta_df_full, features = mdata.load_full_meta(debug=False, signal='blood')
    meta_df_full = mdata.filter_CBC_dataset(meta_df_full, features)
    meta_df_full = meta_df_full.dropna(subset=features)

    # Find subset with longitudinal data
    uids = meta_df_full.groupby('uid').size()[(meta_df_full.groupby('uid').size() > 1)].index.values
    meta_df_full_long = meta_df_full[meta_df_full.uid.isin(uids)].copy()
    print(f"Found {len(uids)} unique animals, with {len(meta_df_full_long)} records")

    # Keep al least 8 mice per strain
    strains = meta_df_full_long.groupby('strain_sex').size()[meta_df_full_long.groupby('strain_sex').size() >= 8].index
    meta_df_full_long = meta_df_full_long[meta_df_full_long.strain_sex.isin(strains)]
    meta_df_full_long = meta_df_full_long.sort_values(['uid', 'age'])
    print(f"Applying strain filter: \n Found {meta_df_full_long['uid'].nunique()} unique animals, "
          f"with {len(meta_df_full_long)} records")

    X_all = meta_df_full[features].values
    X_SSI, Y_SSI, ind = mutils.get_pairs_longitudinal(
        meta_df_full_long, features, None, dt=26.0, dt_tol=0.5, normalize=False, index=True, age_max=80)

    assert config['input_dim'] == X_all.shape[1], \
        f"Input data shape[1]:{X_all.shape[1]} doesn't match model config {config['input_dim']}"
    labels = meta_df_full_long['uid'].values

    tf.keras.backend.clear_session()
    model = mmodel.AE_AR(**config)
    model.train(X_all, X_SSI, Y_SSI, nepoch=args.epoch, batch_size=args.batch_size, lr=args.lr, seed=None,
                groups_ssi=meta_df_full_long.loc[ind[0], 'uid'],
                weight_ae=config['weight_ae'],
                weight_ssi=config['weight_ssi'],
                weight_ssi_raw=config['weight_ssi_raw'],
                verbose=0,
                tbCall=args.tb,
                scaling='stand',
                save_every=args.save_every,
                preproc_func=None,
                save_dir=savedir)
    save_dir = model.save_model(savedir)
