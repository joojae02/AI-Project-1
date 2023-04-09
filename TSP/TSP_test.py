from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import itertools
import networkx as nx

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
    # print(best_path[:])
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
            best_distance = dist
            best_path = path
    return best_path, best_distance



# 군집 수
k = 10

mutation_rate = 0.1

# 도시의 좌표를 생성
cities_coords = []
with open('./2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
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

def total_cal_distance(city_order):
    total_distance = 0
    for i in range(len(city_order)):
        total_distance += distance(cities_coords[city_order[i]], cities_coords[city_order[(i+1) %len(city_order)]])
    return total_distance

# 클러스터 별로 유전 알고리즘 돌리기
total_path = []
best_cluster = []
for i in range(k):
    best_path, best_distance = genetic_algorithm(len(cluster_coords[i]), cluster_coords[i], 100)
    best_cluster.append(best_path)
    total_path.extend(best_path)



# 클러스터로 트리 노드 생성 후 합치기
# def build_tree(best_cluster):
#     tree = {}
    
#     # 각 클러스터에 대해 트리 노드 생성
#     node = tree
#     for city in best_cluster:
#         if city not in node:
#             node[city] = {}
#         node = node[city]
    
#     node["cluster_" + str(i)] = {"path": best_cluster, "cost": total_cal_distance(best_cluster)}
#     # 생성된 트리 업데이트
#     update_tree(tree)
    
#     return tree
# def update_tree(node):
#     # 노드가 leaf 노드인 경우 (클러스터의 최적 경로를 찾은 경우)
#     if "path" in node:
#         return node["cost"]

#     # 노드가 내부 노드인 경우 (재귀적으로 자식 노드를 업데이트하고 최적 경로를 찾음)
#     optimal_cost = float("inf")
#     optimal_path = None
#     for child in node.values():
#         child_cost = update_tree(child)
#         if child_cost < optimal_cost:
#             optimal_cost = child_cost
#             optimal_path = child["path"]

#     # 현재 노드의 최적 경로와 비용 업데이트
#     node["path"] = optimal_path
#     node["cost"] = optimal_cost + total_cal_distance(optimal_path)

#     return node["cost"]

# def merge_clusters(clusters):
#     paths = []
#     for i, cluster in enumerate(clusters):
#         tree = build_tree(cluster)
#         path = get_optimal_path(tree)
#         paths.append(path)
#     final_path = [city for path in paths for city in path]
#     return final_path

# def get_optimal_path(tree):
#     node = tree
#     while "path" not in node:
#         min_child_cost = float("inf")
#         for child in node.values():
#             if child["cost"] < min_child_cost:
#                 min_child_cost = child["cost"]
#                 min_child_path = child["path"]
#         node = next((child for child in node.values() if child["path"] == min_child_path), None)
#     return node["path"]
# final = merge_clusters(best_cluster)


# MST 구성후 합치기
def merge_clusters(clusters):
    # 클러스터 중앙값 기준
    # centers = []
    # for cluster in clusters:
    #     center = np.mean(cluster, axis=0)
    #     centers.append(center)

    # # 클러스터들의 중심점으로부터 가장 가까운 다른 중심점과의 거리를 계산
    # adj_matrix = np.zeros((len(centers), len(centers)))
    # for i, center1 in enumerate(centers):
    #     for j, center2 in enumerate(centers):
    #         if i != j:
    #             distance = np.linalg.norm(center1 - center2)
    #             adj_matrix[i, j] = distance
    distances = []
    for cluster in clusters:
        city_coords = [cities_coords[i] for i in cluster]
        dist = np.linalg.norm(np.array(city_coords)[:, np.newaxis, :] - np.array(city_coords), axis=2)
        distances.append(dist)

    # 인접 행렬 생성
    n = len(clusters)
    adj_matrix = np.zeros((n * len(clusters[0]), n * len(clusters[0])))
    for i in range(n):
        for j in range(i+1, n):
            cluster_i = clusters[i]
            cluster_j = clusters[j]
            city_coords_i = [cities_coords[k] for k in cluster_i]
            city_coords_j = [cities_coords[k] for k in cluster_j]
            for k in range(len(cluster_i)):
                for l in range(len(cluster_j)):
                    adj_matrix[i*len(clusters[0])+k, j*len(clusters[0])+l] = np.linalg.norm(np.array(city_coords_i[k]) - np.array(city_coords_j[l]))
                    adj_matrix[j*len(clusters[0])+l, i*len(clusters[0])+k] = adj_matrix[i*len(clusters[0])+k, j*len(clusters[0])+l]
    
    # MST를 계산하여 반환
    G = nx.from_numpy_array(adj_matrix)
    T = nx.minimum_spanning_tree(G)

    # MST의 간선을 따라 경로 생성
    path = []
    visited = set()
    for u, v in nx.dfs_edges(T):
        if u not in visited:
            if u < len(clusters):
                path.extend(clusters[u])
            visited.add(u)
    if v not in visited:
        if v < len(clusters):
            path.extend(clusters[v])
        visited.add(v)
    # 방문하지 않은 노드들을 추가
    for i, cluster in enumerate(clusters):
        if i not in visited:
            path.extend(cluster)

    return path

def cal_cluster_distance(cluster1, cluster2):
    # 두 클러스터 사이의 거리 계산
    return distance(cluster1[-1], cluster2[0])


# 클러스터들 합치기
final = merge_clusters(best_cluster)

missing = []
for i in range(999):
    if i not in final :
        missing.append(i)
print("missing : ", missing)
print("total distance", total_cal_distance(final))

# 클러스터별로 간선 긋기
# color = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'violet', 'dodgerblue', 'limegreen']
# 
# for c in range(len(best_cluster)) :
#     for i in range(len(best_cluster[c])-1):
#         plt.plot([cities_coords[best_cluster[c][i]][0], cities_coords[best_cluster[c][i+1]][0]], [cities_coords[best_cluster[c][i]][1], cities_coords[best_cluster[c][i+1]][1]], color=color[c])
#     plt.plot([cities_coords[best_cluster[c][-1]][0], cities_coords[best_cluster[c][0]][0]], [cities_coords[best_cluster[c][-1]][1], cities_coords[best_cluster[c][0]][1]], color=color[c])
# plt.show()

for i in range(len(final)-1):
    plt.plot([cities_coords[final[i]][0], cities_coords[final[i+1]][0]], [cities_coords[final[i]][1], cities_coords[final[i+1]][1]], color='k')
plt.plot([cities_coords[final[-1]][0], cities_coords[final[0]][0]], [cities_coords[final[-1]][1], cities_coords[final[0]][1]], color='k')
plt.show()



