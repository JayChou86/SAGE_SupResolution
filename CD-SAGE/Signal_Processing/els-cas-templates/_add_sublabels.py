"""
Add sub-panel labels (a), (b), (c) … to matplotlib axes.

Usage:
    from _add_sublabels import add_sublabels
    add_sublabels([ax1, ax2, ax3])  # adds (a), (b), (c) to top-left
"""
import matplotlib.pyplot as plt


def add_sublabels(axes, fmt='(%s)', fontsize=12, fontweight='bold',
                  offset_x=-0.08, offset_y=1.02, ha='left', va='bottom'):
    """
    Place sub-panel labels at the top-left of each axis.

    Parameters
    ----------
    axes : list of matplotlib Axes
    fmt : str, format string with one %s placeholder for the letter.
    fontsize : int
    fontweight : str
    offset_x, offset_y : float
        Position in axes-fraction coordinates relative to each axis.
    """
    for i, ax in enumerate(axes):
        label = fmt % chr(ord('a') + i)
        ax.text(offset_x, offset_y, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, ha=ha, va=va)


# Example
if __name__ == '__main__':
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.plot([0, 1], [0, 1])
    add_sublabels(axes)
    fig.tight_layout()
    fig.savefig('_sublabel_demo.png', dpi=150)
    print('Saved _sublabel_demo.png')
