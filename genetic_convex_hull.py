import random
import numpy as np
from scipy.spatial import distance
import csv
import math

cities = []
sol = []


#
# 유전 알고리즘 + TSP
# ordered crossover
# 부모 두 개의 순서 유지, 일부 구간을 무작위로 선택 후 유전
# swap mutation
# 랜덤하게 선택된 두 위치의 값을 교환
#
# Partially Mapped Crossover(PMX)나 Order Crossover(OX)
# Swap Mutation나 Inversion Mutation
#
# convex-hull algorithm

# 두 점 사이의 거리 계산 함수
def distance_m(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist


# Convex Hull 알고리즘
def convex_hull(points):
    n = len(points)
    points.sort() # x 좌표 기준 정렬
    lower_hull = [points[0], points[1]] # 하단 볼록 껍질 초기화

    for i in range(2, n):
        lower_hull.append(points[i])
        while len(lower_hull) >= 3 and not is_clockwise(lower_hull[-3], lower_hull[-2], lower_hull[-1]):
            # 마지막 3개 점이 시계 방향이 아니면 마지막 2개 점 제거
            lower_hull.pop(-2)

    upper_hull = [points[-1], points[-2]] # 상단 볼록 껍질 초기화
    for i in range(n-3, -1, -1):
        upper_hull.append(points[i])
        while len(upper_hull) >= 3 and not is_clockwise(upper_hull[-3], upper_hull[-2], upper_hull[-1]):
            # 마지막 3개 점이 시계 방향이 아니면 마지막 2개 점 제거
            upper_hull.pop(-2)

    # 볼록 껍질 합치기
    hull = lower_hull + upper_hull[1:-1]
    return hull


# 시계 방향 여부 판별 함수
def is_clockwise(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) < 0


# TSP 해결 함수
def tsp_convex_hull(points):
    n = len(points)
    # Convex Hull 찾기
    hull = convex_hull(points)

    # Convex Hull 내부의 점들 중 최소 거리 찾기
    min_distance = float('inf')
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            d = distance_m(hull[i], hull[j])
            if d < min_distance:
                min_distance = d
                min_i = i
                min_j = j

    # 최소 거리를 연결하는 경로 찾기
    path = [hull[min_i], hull[min_j]]
    for i in range(n):
        if points[i] not in path:
            if is_clockwise(hull[min_i], points[i], hull[min_j]):
                path.insert(-1, points[i])
            else:
                path.insert(1, points[i])

    return path


# 구간 나누기
def divide_into_segments(points, x_segment_length, y_segment_length):
    # x, y 좌표를 따로 분리
    x_coords, y_coords = zip(*points)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    x_segments = np.arange(min_x, max_x+x_segment_length, x_segment_length)
    y_segments = np.arange(min_y, max_y+y_segment_length, y_segment_length)
    segments = []

    for i in range(len(x_segments) - 1):
        for j in range(len(y_segments) - 1):
            segment = []
            for point in points:
                if x_segments[i] <= point[0] < x_segments[i+1] and \
                   y_segments[j] <= point[1] < y_segments[j+1]:
                    segment.append(point)
            if segment:
                segments.append(segment)

    return segments


#
# 구간별 convex-hull 알고리즘 사용
def find_convex_hulls(segments):
    hulls = []
    for segment in segments:
        hull = tsp_convex_hull(segment)
        hulls.append(hull)
    return hulls

def dict_to_list(tree_dict):
    tree_list = [(k, v) for k, v in tree_dict.items()]
    for i, (node, neighbors) in enumerate(tree_list):
        tree_list[i] = (node, neighbors.copy())  # 새로운 리스트 생성
    return tree_list




def build_tree(segments):
    n = len(segments)
    tree = {0: []}
    best_cost = np.inf

    for i in range(n):
        for j in range(n):
            if i != j:
                segment_i = segments[i]
                segment_j = segments[j]
                cost = distance.cdist(segment_i, segment_j).min()

                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)

    i, j = best_pair
    tree[0].extend([i, j])

    visited = set([i, j])

    def dfs(node):
        nonlocal visited, tree, segments, n

        for neighbor in range(n):
            if neighbor not in visited:
                segment_i = segments[node]
                segment_j = segments[neighbor]
                cost = distance.cdist(segment_i, segment_j).min()

                if cost < np.inf:
                    visited.add(neighbor)
                    if node not in tree:
                        tree[node] = []
                    tree[node].append(neighbor)
                    dfs(neighbor)

    dfs(i)

    return dict_to_list(tree)  # 딕셔너리를 리스트로 변환하여 반환




with open("TSP.csv", mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        tmp = [float(row[0]), float(row[1])]
        cities.append(tmp)
total_cost = 0

num_cities = len(cities)
gene_size = 100
num_generations = 500
mutation_rate = 0.01


# 현재까지의 거리
def total_distance(city_order):
    total_distance = 0
    for i in range(len(city_order)):
        from_city = cities[city_order[i]]
        to_city = cities[city_order[(i + 1) % len(city_order)]]
        total_distance += distance_m (from_city, to_city)
    return total_distance


# 초기 세팅 // 하나의 유전자는 도시 개수 만큼 // 랜덤으로 배치
def create_initial_gene():
    gene = []
    segments = divide_into_segments(cities, 20, 20)
    hulls = find_convex_hulls(segments)
    tree = build_tree(hulls)
    for i in range(gene_size):
        city_order = tree.copy()
        city_order.insert(0, 0)
        gene.append(city_order)
    return gene


# city_order에서 총 거리가 가장 짧은 애들로 2개 선택
def select(gene):
    fitness_list = []
    for i in range (len (gene)):
        city_order = gene[i]
        fitness = total_distance (city_order)
        fitness_list.append ((city_order, fitness))
    fitness_list = sorted (fitness_list, key=lambda x: x[1])
    return fitness_list[0][0], fitness_list[1][0]


# 유전
def crossover(parent1, parent2):
    parent1_length = len (parent1)
    parent2_length = len (parent2)

    start = random.randint (1, len (parent1) - 1)
    end = random.randint (start, len (parent1) - 1)
    child = [-1 for i in range (parent1_length)]
    child[0] = 0
    for i in range (start, end + 1):
        child[i] = parent1[i]
    j = end + 1

    for i in range (1, parent2_length):
        if parent2[i] not in child:
            if j == parent1_length:
                j = 1
            child[j] = parent2[i]
            j += 1
    return child


def mutation(city_order):
    if random.random () < mutation_rate:
        i = random.randint ()
        j = random.randint (0, len (city_order) - 1)
        city_order[i], city_order[j] = city_order[j], city_order[i]
    return city_order


def genetic_algorithm():
    gene = create_initial_gene ()
    best_distance = float ('inf')
    best_city_order = None
    for i in range (num_generations):
        parent1, parent2 = select (gene)
        child = crossover (parent1, parent2)
        child = mutation (child)
        gene = [parent1, parent2, child]

        for j in range (gene_size - 3):
            new_child = crossover (parent1, parent2)
            new_child = mutation (new_child)
            gene.append (new_child)

        fitness_list = []
        for city_order in gene:
            fitness = total_distance (city_order)
            fitness_list.append ((city_order, fitness))
        fitness_list = sorted (fitness_list, key=lambda x: x[1])

        best_route = fitness_list[0]
        if best_route[1] < best_distance:
            best_distance = best_route[1]
            best_city_order = best_route[0]
            print ("///////////////////////////////////////////////////////")
            print (best_distance)  # 적합도 (총 거리)
            print (best_city_order[:10])  # 순서중 10개만 출력
            print ("///////////////////////////////////////////////////////")

    return best_city_order, best_distance


best_ordered, best_distance = genetic_algorithm ()
# print("Best ordered:", best_ordered)
print ("Shortest distance:", best_distance)