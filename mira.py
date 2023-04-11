from sklearn.cluster import KMeans
import numpy as np
import csv
import random
import math


def kmeans_cluster(cities_coords, k):
    # k-means 클러스터링 수행
    kmeans = KMeans(n_clusters=k, random_state=0).fit(cities_coords)
    # 각 도시의 군집 번호 확인
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    # 각 군집에 속한 도시의 x, y 좌표 추출
    cluster_coords = []
    for i in range(k):
        cluster_coords.append(cities_coords[cluster_labels == i])
    return cluster_coords, cluster_centers

# Euclidean distance


def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist


def generate_initial_solution(population_size, cities_coords):
    num_cities = len(cities_coords)
    population = []
    for i in range(population_size):
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


def select(population, cities_coords, k=20):
    participants = random.sample(population, k)
    winner = max(
        participants, key=lambda x: non_circuit_fitness(x, cities_coords))
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
        new_population = []
        elite_size = int(population_size * 0.1)  # keep top 10% of individuals
        new_population.extend(
            sorted(population, key=lambda x: non_circuit_fitness(x, cities_coords))[:elite_size])

        for i in range(population_size - elite_size):
            parent1 = select(population, cities_coords)
            parent2 = select(population, cities_coords)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        new_population = sorted(
            new_population, key=lambda x: non_circuit_fitness(x, cities_coords))
        new_best_path = new_population[0]
        new_best_distance = non_circuit_fitness(new_best_path, cities_coords)
        if new_best_distance < best_distance:
            best_path = new_best_path
            best_distance = new_best_distance
            print("///////////////////////////////////////////////////////")
            print(best_distance)  # 적합도 (총 거리)
            print(best_path[:10], len(best_path))  # 순서중 10개만 출력
            print("///////////////////////////////////////////////////////")

    # 클러스터 내 인덱스에서 전체 인덱스 path로 변경
    for idx, city in enumerate(best_path):
        best_path[idx] = np.where(
            (all_cities_coords == cities_coords[city]).all(axis=1))[0][0]

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


# 군집 수
k = 10

mutation_rate = 0.3

# 도시의 좌표를 생성
all_cities_coords = []
with open('./2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        all_cities_coords.append(list(map(float, row)))

all_cities_coords = np.array(all_cities_coords)

# 클러스터 별 좌표 리스트와 클러스터 중심 좌표 리턴
cluster_coords, cluster_centers = kmeans_cluster(all_cities_coords, k)

# 클러스터 별 유전 알고리즘 돌리고 리스트에 저장
best_cluster = []
for i in range(k):
    best_path, best_distance = genetic_algorithm(50, cluster_coords[i], 1)
    best_cluster.append([best_path, best_distance])

# 클러스터 그냥 이어 붙이기
#res = []
# for i in range(len(best_cluster)):
#    res += best_cluster[i][0]
#print(circuit_fitness(res, all_cities_coords))

# 클러스터 a* search
cluster_astar = []
c_path, c_distance = a_star(0, list(range(k)), cluster_centers)
print(c_path)

# 순서에 따라 클러스터 이어붙이기
res = []
for i in c_path:
    res += best_cluster[i][0]
print(circuit_fitness(res, all_cities_coords))

# 최적해 8등분해서 a* search 해보기 (1000 % 8 == 0, 나누어 떨어져야 모든 경로가 포함된다.)
best_astar = []
for i in range(0, len(res), 7):
    astar_path, astar_distance = a_star(
        res[i], res[i:i+7], all_cities_coords)
    best_astar += astar_path
print(len(best_astar))
print(circuit_fitness(best_astar, all_cities_coords))
