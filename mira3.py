from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math


# Euclidean distance


def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist


def generate_initial_solution(population_size, cities_coords):
    num_cities = len(cities_coords)
    population = []
    for i in range(population_size):
        print("generation", i)
        start_city = random.randint(0, num_cities-1)
        unvisited_cities = list(range(num_cities))
        unvisited_cities.remove(start_city)
        solution = [start_city]

        while unvisited_cities:
            last_city = solution[-1]
            distances = [distance(cities_coords[last_city], cities_coords[j])
                         for j in unvisited_cities]
            next_city = unvisited_cities[np.argmin(distances)]
            unvisited_cities.remove(next_city)
            solution.append(next_city)

        population.append(solution)

    return population

# fitness


def circuit_fitness(path, cities_coords):
    total_dist = 0
    for i in range(len(path)):
        pos_city_1 = cities_coords[path[i]]
        pos_city_2 = cities_coords[path[(i+1) % len(path)]]
        dist = distance(pos_city_1, pos_city_2)
        total_dist += dist
    return total_dist


def non_circuit_fitness(path, cities_coords):
    total_dist = 0
    for i in range(len(path) - 1):
        pos_city_1 = cities_coords[path[i]]
        pos_city_2 = cities_coords[path[i+1]]
        dist = distance(pos_city_1, pos_city_2)
        total_dist += dist
    return total_dist

# select


def select(population, cities_coords, k=10):
    participants = random.sample(population, k)
    winner = min(
        participants, key=lambda x: circuit_fitness(x, cities_coords))
    return winner

# crossover


def crossover(parent1, parent2):
    parent_length = len(parent1)
    start, end = sorted([random.randint(0, parent_length-1) for _ in range(2)])
    child = [-1 for _ in range(parent_length)]

    # 유지해야 할 구간을 parent1에서 가져옴
    child[start:end+1] = parent1[start:end+1]

    # parent2에서 순서를 따르면서 child의 빈 칸을 채움
    j = end+1
    for i in range(parent_length):
        if parent2[i] not in child:
            if j == parent_length:
                j = 0
            child[j] = parent2[i]
            j += 1

    return child


def mutate(path):
    if random.random() < mutation_rate:
        i = random.randint(0, len(path)-1)
        j = random.randint(0, len(path)-1)
        path[i], path[j] = path[j], path[i]
    return path


def genetic_algorithm(population_size, cities_coords, iteration):
    global all_cities_coords
    population = generate_initial_solution(population_size, cities_coords)
    best_path = []
    best_distance = float('inf')
    for i in range(iteration):
        print(i)
        new_population = []
        elite_size = int(population_size * 0.1)  # keep top 10% of individuals
        new_population.extend(
            sorted(population, key=lambda x: circuit_fitness(x, cities_coords))[:elite_size])

        for i in range(population_size - elite_size):
            parent1 = select(population, cities_coords)
            parent2 = select(population, cities_coords)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        new_population = sorted(
            new_population, key=lambda x: circuit_fitness(x, cities_coords))
        new_best_path = new_population[0]
        new_best_distance = circuit_fitness(new_best_path, cities_coords)
        if new_best_distance < best_distance:
            best_path = new_best_path
            best_distance = new_best_distance
            print("///////////////////////////////////////////////////////")
            print(best_distance)  # 적합도 (총 거리)
            print(best_path[:10], len(best_path))  # 순서중 10개만 출력
            print("///////////////////////////////////////////////////////")

    return best_path, best_distance


def tsp_heuristic(state, remaining_cities, all_cities_coords):
    # If all cities have been visited, the minimum cost to visit all remaining cities is zero.
    if not remaining_cities:
        return 0

    # Calculate the minimum cost to visit all remaining cities.
    min_cost = math.inf
    for city in remaining_cities:
        cost = distance(all_cities_coords[state[-1]], all_cities_coords[city])
        new_state = state + [city]
        new_remaining_cities = [c for c in remaining_cities if c != city]
        cost += tsp_heuristic(new_state,
                              new_remaining_cities, all_cities_coords)
        if cost < min_cost:
            min_cost = cost

    # Return the estimated minimum cost.
    return min_cost


def a_star(start, cities, all_cities_coords):
    visited = [start]
    path = [start]
    remaining_cities = [c for c in cities if c not in visited]
    best_dist = 0

    while remaining_cities:
        nearest_city = None
        nearest_dist = float('inf')
        heuristic = None
        for i in remaining_cities:
            dist = distance(all_cities_coords[path[-1]], all_cities_coords[i])
            heuristic = tsp_heuristic(
                visited, remaining_cities, all_cities_coords)
            if dist + heuristic < nearest_dist:
                nearest_city = i
                nearest_dist = dist + heuristic
        visited.append(nearest_city)
        path.append(nearest_city)
        remaining_cities.remove(nearest_city)
        best_dist += nearest_dist - heuristic
    return path, best_dist


mutation_rate = 0.1

all_cities_coords = []
with open('./2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        all_cities_coords.append(list(map(float, row)))

best_path, best_distance = genetic_algorithm(20, all_cities_coords, 50)

for i in range(1000):
    start_index = random.randint(0, len(best_path)-5)
    end_index = start_index + 5
    temp = best_path[start_index:end_index]

    coords = []
    for i in temp:
        coords.append(all_cities_coords[i])

    res_path = list(range(5))
    for i in range(5):
        path, dist = a_star(i, list(range(5)), coords)
        if non_circuit_fitness(path, coords) < non_circuit_fitness(res_path, coords):
            res_path = path

    global_res_path = []
    for i in range(5):
        idx = res_path[i]
        global_res_path.append(temp[idx])

new_best_path = best_path[:start_index] + \
    global_res_path[:] + best_path[end_index:]
if circuit_fitness(best_path, all_cities_coords) > circuit_fitness(new_best_path, all_cities_coords):
    best_path = new_best_path
print(circuit_fitness(best_path, all_cities_coords))

for i in range(len(best_path)-1):
    plt.plot([all_cities_coords[best_path[i]][0], all_cities_coords[best_path[i+1]][0]],
             [all_cities_coords[best_path[i]][1], all_cities_coords[best_path[i+1]][1]], color='b')
plt.plot([all_cities_coords[best_path[-1]][0], all_cities_coords[best_path[0]][0]],
         [all_cities_coords[best_path[-1]][1], all_cities_coords[best_path[0]][1]], color='b')
plt.show()
