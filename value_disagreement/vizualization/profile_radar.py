import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from value_disagreement.extraction import ValueConstants


def plot_radar(values, data, outname="test.png", value_type="schwartz"):
    if value_type == "schwartz":
        # Desired circumplex order
        values = [ValueConstants.SCHWARTZ_VALUES[x] for x in ValueConstants.SCHWARTZ_VALUES_CIRCUMPLEX_ORDER]
        values = values[::-1]
    elif value_type == "mft":
        # TODO: ??
        pass
    else:
        raise ValueError(f"Unknown value type ({value_type}) to plot profile for")
    spoke_labels = values
    N = len(spoke_labels)
    theta = radar_factory(N, frame='circle')

    fig = plt.figure(figsize=(9, 9))
    ax = plt.gca(projection='radar')
    fig.subplots_adjust(wspace=0.25, hspace=0.6, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    title, plot_data = data

    # plot data is in alphabetical value order
    plot_data = np.array(plot_data)
    # Get circumplex order (however, matplotlib will put it counter clockwise)
    circumplex_order_data = plot_data[:, ValueConstants.SCHWARTZ_VALUES_CIRCUMPLEX_ORDER]
    # Reverse order to make it clockwise
    cod_reversed = circumplex_order_data[:, ::-1].tolist()

    # Plot it!
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='large', position=(0.5, 1.5), horizontalalignment='center')
    for d, color in zip(cod_reversed, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    plt.savefig(outname)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    From https://matplotlib.org/3.5.0/gallery/specialty_plots/radar_chart.html

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    values = ["achievement","benevolence","conformity","hedonism","power","security","self-direction","stimulation","tradition","universalism"]
    data = ('Schwartz values', [[0.88, 0.11, 0.23, 0.33, 0.40, 0.56, 0.61, 0.70, 0.80, 0.9]])
    return values, data


if __name__ == '__main__':
    values, data = example_data()
    plot_radar(values, data)
