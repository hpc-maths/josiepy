import matplotlib.pyplot as plt

from josie.geom import Line, CircleArc
from josie.mesh import Mesh


def test_mesh():
    left = Line([0, 0], [0, 1])
    bottom = CircleArc([0, 0], [1, 0], [0.5, 0.5])
    right = Line([1, 0], [1, 1])
    top = Line([1, 1], [0, 1])

    mesh = Mesh(left, bottom, right, top)
    x, y = mesh.generate(20, 20)

    plt.plot(x, y, 'ko')
    # left.plot()
    # bottom.plot()
    # right.plot()
    # top.plot()
    plt.show()
