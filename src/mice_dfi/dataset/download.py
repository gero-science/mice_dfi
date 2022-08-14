import os
import wget
from tqdm.auto import tqdm

_datasets = (
    ('Peters1', 'CBC'),
    ('Peters2', 'CBC'),
    ('CGDpheno1', 'CBC'),
    ('Svenson3', 'CBC'),
    ('CGDpheno3', 'CBC'),
    ('Justice2', 'CBC'),
    ('Jaxpheno4', 'CBC'),
    ('Petkova1', 'CBC'),
    ('HMDPpheno5', 'CBC'),
    ('Lake1', 'CBC'),
    ('Peters4', 'CBC'),
    ('Yuan1', 'serum'),
    ('Ackert1', 'bone'),
    ('Seburn2', 'gait'),
    ('Yuan3', 'serum'),
    ('Yuan2', 'lifespan'),
)
_datasets_misc = {
    'Yuan2_strainmeans': ('https://mpdpreview.jax.org/projects/Yuan2/csvstrainmeans', 'lifespan'),
}

_dataset_ipt1 = {
    'url': 'https://phenomedoc.jax.org/ITP1/Data_frozen_public/Lifespan_C{:s}.xlsx',
    'path': 'lifespan',
    'years': ['2004', '2005', '2006', '2007', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
}

_mpd_url = 'https://phenomedoc.jax.org/MPD_projdatasets/{:s}.csv'

if __name__ == '__main__':

    # # Download main datasets
    for (dataset, path) in tqdm(_datasets):
        os.makedirs('raw/'+path, exist_ok=True)
        filename = wget.download(_mpd_url.format(dataset), out='raw/{:s}/{:s}.csv'.format(path, dataset))

    # # Download misc datasets
    for dataset, (url, path) in tqdm(_datasets_misc.items()):
        os.makedirs('raw/'+path, exist_ok=True)
        filename = wget.download(url, out='raw/{:s}/{:s}.csv'.format(path, dataset))

    # Download ITP1: Interventions Testing Program
    # https://phenome.jax.org/projects/ITP1
    os.makedirs('raw/{:s}/itp1'.format(_dataset_ipt1['path']), exist_ok=True)
    for year in tqdm(_dataset_ipt1['years']):
        wget.download(_dataset_ipt1['url'].format(year), out='raw/{:s}/itp1/'.format(_dataset_ipt1['path']))
