__all__ = [
    'set_style'
]

import matplotlib
import matplotlib.pyplot as plt
import cycler


# If Arial is not found, copy it manually to mpl folder and clear cache
# sudo apt-get install ttf-mscorefonts-installer
# cp /usr/share/fonts/truetype/msttcorefonts/Arial* $VENV_PATH/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/
# rm -r ~/.cache/matplotlib

def set_style(production=False):
    # Set default parameters
    if production:
        scale = 0.5
    else:
        scale = 1.

    matplotlib.rcdefaults()
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'arial'
    plt.rcParams["svg.fonttype"] = 'none'
    plt.rcParams["savefig.pad_inches"] = 0.1*scale

    # Axes
    plt.rcParams['axes.facecolor'] = '#ffffff'
    if production:
        plt.rcParams['axes.labelsize'] = 7
    else:
        plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.labelcolor'] = 'k'
    if production:
        plt.rcParams['axes.titlesize'] = 7
    else:
        plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.linewidth'] = 1.0*scale  # edge linewidth
    plt.rcParams["axes.labelpad"] = 2 * scale

    # Ticks
    if production:
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7
    else:
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.major.width'] = 1.80*scale
    plt.rcParams['ytick.major.width'] = 1.80*scale
    plt.rcParams['xtick.minor.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.major.pad'] = 2.5 * scale
    plt.rcParams['xtick.major.pad'] = 2.5 * scale

    plt.rcParams['xtick.color'] = 'k'
    plt.rcParams['ytick.color'] = 'k'

    # Figure
    plt.rcParams['figure.facecolor'] = '#ffffff'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

    # Lines
    plt.rcParams['lines.linewidth'] = 1.80*scale

    # Legend
    plt.rcParams['legend.frameon'] = False
    if production:
        plt.rcParams['legend.fontsize'] = 7
    else:
        plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['lines.markersize'] = 10 * scale
    plt.rcParams['legend.handletextpad'] = 0.6 * scale


    # Set colormap
    plt.rcParams['image.cmap'] = 'coolwarm'

    color_cycler = cycler.cycler('color', matplotlib.cm.tab10.colors)
    plt.rc('axes', prop_cycle=color_cycler)
