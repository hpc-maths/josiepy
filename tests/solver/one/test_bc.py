from josie.mesh.cell import NeighbourCell


def test_periodic(solver):
    """ Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """

    for left_cell in solver.mesh.cells[0, :]:
        assert isinstance(left_cell.w, NeighbourCell)

    for btm_cell in solver.mesh.cells[:, 0]:
        assert btm_cell.s is None

    for right_cell in solver.mesh.cells[-1, :]:
        assert isinstance(right_cell.e, NeighbourCell)

    for top_cell in solver.mesh.cells[:, -1]:
        assert top_cell.n is None
