from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def legend_workaround(poly3d: Poly3DCollection):
    """
    Workaround the bug on 3D legend causing the exception:
    > AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'

    This function should be removed when the issue is resolved.

    more:
    - https://stackoverflow.com/a/54994985
    - https://github.com/matplotlib/matplotlib/issues/4067
    """
    poly3d._facecolors2d=poly3d._facecolors3d
    poly3d._edgecolors2d=poly3d._edgecolors3d
