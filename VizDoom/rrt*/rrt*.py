import numpy as np
import matplotlib.pyplot as plt
from utils import angle, distance
from utils import node as node
import cv2
import os

dirname = os.path.dirname(__file__)


class RRTStar:
    def __init__(
        self,
        automap,
        start,
        goal,
        dist,
        obstacle_color=[11, 27, 47],
        max_iterations=10000,
        goal_radius=10,
    ):
        """Initialising the Path Finder

        :param np.ndarray automap: Image used for computing path
        :param node start: starting node
        :param node goal: target node
        :param float dist: distance for each edge
        :param list obstacle_color: color correspoinding to the obstacle in the provided automap
        :param int max_iterations: maximum iterations for guessing the next node
        :param float goal_radius
        """

        self.automap = automap
        self.start = start
        self.goal = goal
        self.dist = dist
        self.obstacle_color = obstacle_color
        self.max_iterations = max_iterations
        self.goal_radius = goal_radius
        self.tree = [self.start]
        self.start.cost = 0
        self.temp = self.automap.copy()
        self.iterations = 0

    def detect_obstacle(self, node1, node2):
        """Fuction for Detecting obstacle between two given nodes

        :param node node1
        :param node node2

        :return: True if Obstacle is detected otherwise False
        :rtype: bool
        """

        """Creating the list of pixels between node1 and node2"""
        if node2.x != node1.x:
            delx = abs(node2.x - node1.x)
            dely = abs(node2.y - node1.y)
            delmax = max(delx, dely)
            f = (
                lambda x: (node2.y - node1.y) * (x - node1.x) / (node2.x - node1.x)
                + node1.y
            )
            x = (
                np.linspace(node1.x, node2.x, delmax + 2)
                if node2.x > node1.x
                else np.linspace(int(node2.x), int(node1.x), delmax + 2)
            )
            y = f(x)

        else:
            """Creating a virticle line"""
            y = (
                np.arange(node2.y, node1.y)
                if node2.y < node1.y
                else np.arange(node1.y, node2.y)
            )
            x = np.full(shape=y.shape, fill_value=node1.x)

        for a, b in zip(x, y):
            a, b = round(a), round(b)
            if 10 < a < self.automap.shape[0] and 10 < b < self.automap.shape[1] - 10:
                # print(a, b)
                color = self.automap[a, b]
                upcolor = self.automap[a - 1, b]
                if any(color == self.obstacle_color) or any(
                    upcolor == self.obstacle_color
                ):
                    return True
            else:
                return True
        return False

    def random_node(self):
        """Choosing a random node in the automap

        :return: random_node
        :rtype: node
        """
        m = np.random.randint(0, self.automap.shape[0])
        n = np.random.randint(0, self.automap.shape[1])
        rand_node = node(x=m, y=n)
        for tree_node in self.tree:
            if rand_node.x == tree_node.x and rand_node.y == tree_node.y:
                return False
        else:
            return rand_node

    def find_closest(self, n1):
        """Finding the closest node in the current tree

        :param node n1: node for which the closest node is to be computed

        :return: closest node
        :rtype: node
        """

        m = np.inf
        closest = None
        for i, n in enumerate(self.tree):
            d = distance(n, n1)
            if d < m:
                m = d
                closest = i
        return closest

    def node_at_k(self, n1, n2):
        """Finding a node at a distance of self.dist from the closest node in the direction of the new random node
        :param node n1: closest node
        :param node n2: random node

        :return: node at a distance self.dist from n1
        :rtype: node
        """
        n_cap = [(n2.x - n1.x), (n2.y - n1.y)] / distance(n1, n2)
        new_node = node(0, 0)
        new_node.x = n1.x + int(n_cap[0] * self.dist)
        new_node.y = n1.y + int(n_cap[1] * self.dist)
        d = distance(n1, new_node)
        return new_node, d

    def rewire(self, new_node, r=100):
        for n in self.tree:
            d = distance(new_node, n)
            if (
                d < r
                and n.cost + d < new_node.cost
                and not self.detect_obstacle(n, new_node)
            ):
                new_node.parent = n
                new_node.cost = n.cost + d
        self.tree.append(new_node)

    def findpath(self):
        """Finding path from self.start to self.goal"""

        while self.iterations < self.max_iterations:
            rand_node = self.random_node()
            if not rand_node:
                continue
            closest = self.tree[self.find_closest(rand_node)]
            new_node, new_cost = self.node_at_k(closest, rand_node)
            if not self.detect_obstacle(closest, new_node):
                new_node.parent = closest
                new_node.cost = new_node.parent.cost + new_cost
                self.rewire(new_node)
                self.iterations += 1
                cv2.line(
                    self.temp,
                    (new_node.parent.y, new_node.parent.x),
                    (new_node.y, new_node.x),
                    (200, 200, 200),
                )
                cv2.imshow("map", self.temp)
                if cv2.waitKey(1) == ord("q"):
                    return

            if distance(new_node, self.goal) < self.goal_radius:
                temp2 = self.automap.copy()
                print("FOUND!!!")
                self.goal.parent = new_node
                path_node = self.goal
                self.goal
                while path_node.parent is not None:
                    cv2.line(
                        temp2,
                        (path_node.y, path_node.x),
                        (path_node.parent.y, path_node.parent.x),
                        (0, 255, 0),
                    )
                    cv2.imshow("map", temp2)
                    if cv2.waitKey(1) == ord("q"):
                        return
                    path_node = path_node.parent
                cv2.imwrite(os.path.join(dirname, "final_path.png"), temp2)
                cv2.waitKey(0)
                return


def process_image(img):
    start = None
    goal = None
    obstacles = []
    for i in range(0, img.shape[0]):
        for j in range(img.shape[1]):
            if all(img[i, j] == [255, 255, 255]) and start is None:
                start = node(i, j)
            if all(img[i, j] == [255, 0, 0]) and goal is None:
                goal = node(i, j)
            if start is not None and goal is not None:
                break
    return start, goal


if __name__ == "__main__":
    im = cv2.imread(os.path.join(dirname, "map_full.png"))
    start, goal = process_image(im)
    rrtstarer = RRTStar(im, start, goal, 7, [11, 27, 47], goal_radius=15)
    rrtstarer.findpath()
