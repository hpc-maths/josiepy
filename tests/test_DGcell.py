from josie.mesh.cell import DGCell
import numpy as np

# Compare the mass and stiff matrices in the element of reference [-1,1]x[-1,1]


def test_mass_stiff_matrix():
    Mref = DGCell.refMass()
    # Sref = DGCell.refStiff(M=Mref)
    MrefEdge_tab = DGCell.refMassEdge()

    print(MrefEdge_tab[0])
    print(MrefEdge_tab[1])
    print(MrefEdge_tab[2])
    print(MrefEdge_tab[3])

    # print(Sref)

    M = np.array(
        [
            [4 / 9, 2 / 9, 2 / 9, 1 / 9],
            [2 / 9, 4 / 9, 1 / 9, 2 / 9],
            [2 / 9, 1 / 9, 4 / 9, 2 / 9],
            [1 / 9, 2 / 9, 2 / 9, 4 / 9],
        ]
    )

    assert np.linalg.norm(M - Mref) < 1e-14
