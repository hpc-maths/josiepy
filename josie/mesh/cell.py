class GhostCell:
    def __init__(self, value):
        self.new = value


class Cell(GhostCell):
    def __init__(self, p0, p1, p2, p3, i, j, value=0):
        self.centroid = (
            (p0[0] + p1[0] + p2[0] + p3[0])/4,
            (p0[1] + p1[1] + p2[1] + p3[1])/4,
        )
        self.i = i
        self.j = j

        super().__init__(value)
