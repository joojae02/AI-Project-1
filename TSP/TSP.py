from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import itertools


def kmeans_cluster(city_coords, k):
    # k-means 클러스터링 수행
    kmeans = KMeans(n_clusters=k, random_state=0).fit(city_coords)
    # 각 도시의 군집 번호 확인
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    # 각 군집에 속한 도시의 x, y 좌표 추출
    cluster_coords = []
    for i in range(k):
        cluster_coords.append(city_coords[cluster_labels == i])
    return cluster_coords, cluster_centers

# Euclidean distance


def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist

# 초기 해 생성


def generate_initial_solution(population_size, city_coords):
    num_cities = len(city_coords)
    population = []
    for i in range(population_size):
        start_city = random.randint(0, num_cities-1)
        unvisited_cities = list(range(num_cities))
        unvisited_cities.remove(start_city)
        solution = [start_city]

        while unvisited_cities:
            last_city = solution[-1]
            distances = [distance(city_coords[last_city], city_coords[j])
                         for j in unvisited_cities]
            next_city = unvisited_cities[np.argmin(distances)]
            unvisited_cities.remove(next_city)
            solution.append(next_city)

        population.append(solution)

    return population


# fitness

def fitness(path, city_coords):
    total_dist = 0
    for i in range(len(path)):
        pos_city_1 = city_coords[path[i]]
        pos_city_2 = city_coords[path[(i+1) % len(path)]]
        total_dist += distance(pos_city_1, pos_city_2)
    return total_dist

# select


def select(population, city_coords, k=5):
    participants = random.sample(population, k)
    winner = max(participants, key=lambda x: fitness(x, city_coords))
    return winner

# crossover


def crossover(parent1, parent2):
    parent1_length = len(parent1)
    parent2_length = len(parent2)

    start = random.randint(1, len(parent1)-1)
    end = random.randint(start, len(parent1)-1)
    child = [-1 for i in range(parent1_length)]
    for i in range(start, end+1):
        child[i] = parent1[i]
    j = end+1

    for i in range(1, parent2_length):
        if parent2[i] not in child:
            if j == parent1_length:
                j = 1
            child[j] = parent2[i]
            j += 1
    return child

# mutation


def mutate(path):
    if random.random() < mutation_rate:
        i = random.randint(0, len(path)-1)
        j = random.randint(0, len(path)-1)
        path[i], path[j] = path[j], path[i]
    return path

# genetic-algorithm


def genetic_algorithm(population_size, city_coords, iteration):
    global cities_coords
    population = generate_initial_solution(population_size, city_coords)
    best_path = []
    best_distance = float('inf')
    for i in range(iteration):
        new_population = []
        elite_size = int(population_size * 0.1)  # keep top 10% of individuals
        new_population.extend(
            sorted(population, key=lambda x: fitness(x, city_coords))[:elite_size])

        for i in range(population_size - elite_size):
            parent1 = select(population, city_coords)
            parent2 = select(population, city_coords)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        fitness_list = []
        for path in new_population:
            distance = fitness(path, city_coords)
            fitness_list.append([path, distance])

        fitness_list = sorted(fitness_list, key=lambda x: x[1])
        best = fitness_list[0]
        if best[1] < best_distance:
            best_distance = best[1]
            best_path = best[0]
            print("///////////////////////////////////////////////////////")
            print(best_distance)  # 적합도 (총 거리)
            print(best_path[:10], len(best_path))  # 순서중 10개만 출력
            print("///////////////////////////////////////////////////////")
    for idx, coord in enumerate(best_path):
        best_path[idx] = np.where(
            (cities_coords == city_coords[coord]).all(axis=1))[0][0]
    print(best_path[:])
    return best_path, best_distance


def tsp_brute_force(city_center_coords):
    n = len(city_center_coords)
    best_distance = float('inf')
    best_path = None
    for path in itertools.permutations(range(n)):
        dist = 0
        for i in range(n - 1):
            dist += distance(city_center_coords[i],
                             city_center_coords[i+1])
        dist += distance(city_center_coords[n-1],
                         city_center_coords[0])
        if dist < best_distance:
            best_dist = distance
            best_path = path
    return best_path, best_distance


# 군집 수
k = 10

mutation_rate = 0.1

# 도시의 좌표를 생성
cities_coords = []
with open('./TSP/2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities_coords.append(list(map(float, row)))

cities_coords = np.array(cities_coords)

cluster_coords, cluster_centers = kmeans_cluster(cities_coords, k)

# 군집별 scatter plot 그리기
# for i in range(k):
#    plt.scatter(cluster_coords[i][:, 0],
#                cluster_coords[i][:, 1], s=5)
# plt.show()
best_cluster = []
for i in range(k):
    best_path, best_distance = genetic_algorithm(100, cluster_coords[i], 1)
    best_cluster.append([best_path, best_distance])


best_path_cluster, best_distance_cluster = tsp_brute_force(cluster_centers)
print(best_path[:])

res_path = []
res_distance = sum(best_cluster[:][1])
for i in range(k):
    if i == k-1:
        break
    res_distance += distance(cities_coords[best_cluster[best_path_cluster[i]][0][-1]],
                             cities_coords[best_cluster[best_path_cluster[i+1]][0][0]])
    res_path += best_cluster[best_path_cluster[i]][0]

print(len(res_path))
print("best distance: ", res_distance)
