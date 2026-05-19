# -*- coding: utf-8 -*-
# shared_utils/paths.py
# Path management for all variants

#%% Import packages

from pathlib import Path


#%% Constants

# Project root is one level up from shared_utils/
PROJECT_DIR = Path(__file__).resolve().parent.parent

VARIANTS = {
    'single':       'single',
    'coop_partial': 'coop_partial',
    'coop_full':    'coop_full',
    'coop_mixed':   'coop_mixed',
}


#%% Functions

def get_paths(variant):
    '''Return a dict of directories for a given variant.

    Keys: models_dir, plots_dir, videos_dir, logs_dir
    All directories are created if they don't exist.

    Layout:
      single        -> single_agent/{Models, Media/Plots, Media/Videos, Logs}
      coop_partial  -> multi_agent/{Models, Media, Logs}/coop_partial/...
      coop_full     -> multi_agent/{Models, Media, Logs}/coop_full/...
      coop_mixed    -> multi_agent/{Models, Media, Logs}/coop_mixed/...
    '''
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}. Choose from: {list(VARIANTS.keys())}')

    if variant == 'single':
        base = PROJECT_DIR / 'single_agent'
        paths = {
            'models_dir': base / 'Models',
            'plots_dir':  base / 'Media' / 'Plots',
            'videos_dir': base / 'Media' / 'Videos',
            'logs_dir':   base / 'Logs',
        }
    else:
        tag = VARIANTS[variant]
        base = PROJECT_DIR / 'multi_agent'
        paths = {
            'models_dir': base / 'Models' / tag,
            'plots_dir':  base / 'Media' / tag / 'Plots',
            'videos_dir': base / 'Media' / tag / 'Videos',
            'logs_dir':   base / 'Logs' / tag,
        }

    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)

    return paths

#%% End of Script