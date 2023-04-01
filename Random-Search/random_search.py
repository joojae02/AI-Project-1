import numpy as np
import csv
import random

# Euclidean distance measuring function


def distance(x, y):
    dist = np.linalg.norm(np.array(x)-np.array(y))
    return dist


def tsp_cost(path, cities):
    total_cost = 0

    for idx in range(len(path)-1):
        # get city positions
        pos_city_1 = [float(cities[path[idx]][0]), float(cities[path[idx]][1])]
        pos_city_2 = [float(cities[path[idx+1]][0]),
                      float(cities[path[idx+1]][1])]

        # distance calculation
        dist = distance(pos_city_1, pos_city_2)

        # accumulation
        total_cost += dist

    return total_cost


def random_search(cities, iterations):
    best_path = None
    best_cost = float('inf')

    for i in range(iterations):
        # get random sequence
        path = list(range(len(cities)))
        random.shuffle(path)

        # reordering sequence
        idx = path.index(0)

        front = path[idx:]
        back = path[0:idx]

        path = front + back

        # expand 0 city (start) for simplicity
        path.append(int(0))

        # evaluate cost
        cost = tsp_cost(path, cities)

        # update best cost
        if cost < best_cost:
            best_path = path
            best_cost = cost

    return best_path, best_cost


# given cities
cities = []
# solution
sol = []

# get TSP city map
with open('./Random-Search/TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    # read TSP city map
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(row)

# random search
sol, best_cost = random_search(cities, 1000)

# write solution sequence
with open('./Random-Search/solution.csv', mode='w', newline='', encoding='utf-8-sig') as solution:
    writer = csv.writer(solution)
    for row in sol:
        writer.writerow([row])

print('final cost: ' + str(best_cost))
