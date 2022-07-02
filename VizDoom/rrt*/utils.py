import numpy as np


class node:
    def __init__(self, x=None, y=None, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = np.inf

    def angle():
        self.delta = angle(self.parent, self, self.next_node)


def distance(n1: node, n2: node):
    return np.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)


def angle(n1, n2, n3):
    return np.arccos(
        ((n1.x - n2.x) * (n3.x - n2.x) + (n1.y - n2.y) * (n3.y - n2.y))
        / (distance(n1, n2) * distance(n3, n2))
    )
