from heapq import heapify, heappush, heappop
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/VENKATESH/Desktop/occupancy_map.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:/Users/VENKATESH/Desktop/occupancy_map.png")

occupancy_grid = (np.asarray(img) > 0).astype(int)
occupancy_grid_2 = (np.asarray(img2) > 0).astype(int)

def d(v1, v2):
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    distance = math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    return distance


def Neighbours(v):
    global occupancy_grid
    free = []
    x = int(v[0])
    y = int(v[1])

    for i in range(0, 2):
        for j in range(0, 2):
            if 0 <= x + i <= 680 and 0 <= y + j <= 623:
                if occupancy_grid[x + i][y + j] == 1:
                    free.append((x + i, y + j))
            if 0 <= x - i <= 680 and 0 <= y - j <= 623:
                if occupancy_grid[x - i][y - j] == 1:
                    free.append((x - i, y - j))
            if occupancy_grid[x + i][y - j] == 1:
                free.append((x + i, y - j))
            if occupancy_grid[x - i][y + j] == 1:
                free.append((x - i, y + j))
    return free


def RECOVERPATH(s, g, pred):
    global img2
    Q = [g]
    new = g
    img2[g[0], g[1]] = [255, 0, 0]
    while new != s:
        new = pred[new[0]][new[1]]
        img2[new[0], new[1]] = [255, 0, 0]
        Q.append(new)
    Q.reverse()
    return Q


def AStar(s, g):
    Cost = [[float('inf') for i in range(0, 624)] for j in range(0, 681)]
    TCost = [[float('inf') for i in range(0, 624)] for j in range(0, 681)]
    pred = [[float('inf') for i in range(0, 624)] for j in range(0, 681)]

    Cost[s[0]][s[1]] = 0
    TCost[s[0]][s[1]] = d(s, g)
    heap = []
    heapify(heap)
    heappush(heap, (TCost[s[0]][s[1]], s))

    while heap:
        tempv = heappop(heap)
        v = tempv[1]
        if v == g:
            return RECOVERPATH(s, g, pred), TCost[g[0]][g[1]]
        vnbr = Neighbours(v)
        for i in vnbr:
            pvi = Cost[v[0]][v[1]] + d(v, i)
            if pvi < Cost[i[0]][i[1]]:
                pred[i[0]][i[1]] = v
                Cost[i[0]][i[1]] = pvi
                TCost[i[0]][i[1]] = pvi + d(i, g)
                flag = 0
                for q in heap:
                    t = q[1]
                    if t == i:
                        flag = 1
                        del heap[heap.index(q)]
                        heappush(heap, (TCost[i[0]][i[1]], i))
                if flag == 0:
                    heappush(heap, (TCost[i[0]][i[1]], i))
    return []

s=(635,140)
g=(350,400)
L, cost = AStar(s, g)
print(cost)
imgplot = plt.imshow(img2)
plt.savefig("Astar.png",dpi=1200)
plt.show()
