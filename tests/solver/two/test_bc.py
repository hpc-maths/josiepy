from josie.mesh.cell import NeighbourCell


def test_periodic(solver):
    """ Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """

    for left_cell in solver.mesh.cells[0, :]:
        assert isinstance(left_cell.w, NeighbourCell)
        assert left_cell.w.cell == solver.mesh.cells[-1, left_cell.j]

    for btm_cell in solver.mesh.cells[:, 0]:
        assert isinstance(btm_cell.s, NeighbourCell)
        assert btm_cell.s.cell == solver.mesh.cells[btm_cell.i, -1]

    for right_cell in solver.mesh.cells[-1, :]:
        assert isinstance(right_cell.e, NeighbourCell)
        assert right_cell.e.cell == solver.mesh.cells[0, right_cell.j]

    for top_cell in solver.mesh.cells[:, -1]:
        assert isinstance(top_cell.n, NeighbourCell)
        assert top_cell.n.cell == solver.mesh.cells[top_cell.i, 0]
