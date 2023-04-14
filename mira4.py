from sklearn.cluster import KMeans
import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

# 도시의 좌표를 생성
all_cities_coords = []
with open('./2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        all_cities_coords.append(list(map(float, row)))

all_cities_coords = np.array(all_cities_coords)

# 두 도시 사이 거리


def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist

# best_astar순서대로 도시간 그래프 그리기


def draw_plt(best_astar, all_cities_coords):
    for i in range(len(best_astar)-1):
        plt.plot([all_cities_coords[best_astar[i]][0], all_cities_coords[best_astar[i+1]][0]],
                 [all_cities_coords[best_astar[i]][1], all_cities_coords[best_astar[i+1]][1]], color='k')
    plt.plot([all_cities_coords[best_astar[-1]][0], all_cities_coords[best_astar[0]][0]],
             [all_cities_coords[best_astar[-1]][1], all_cities_coords[best_astar[0]][1]], color='k')
    plt.show()

# path의 총 cost


def circuit_fitness(path, cities_coords):
    total_dist = 0
    for i in range(len(path)):
        pos_city_1 = cities_coords[path[i]]
        pos_city_2 = cities_coords[path[(i+1) % len(path)]]
        dist = distance(pos_city_1, pos_city_2)
        total_dist += dist
    return total_dist


class Cluster:
    def __init__(self, cities_coords, k):
        self.cities_coords = cities_coords
        self.k = k
        self.cluster_coords = None
        self.cluster_centers = None
        self.kmeans_cluster()

    def kmeans_cluster(self):
        # k-means 클러스터링 수행
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(
            self.cities_coords)
        # 각 도시의 군집 번호 확인
        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        # 각 군집에 속한 도시의 x, y 좌표 추출
        self.cluster_coords = []
        for i in range(self.k):
            self.cluster_coords.append(
                self.cities_coords[self.cluster_labels == i])

    def get_cluster_coords(self):
        return self.cluster_coords

    def get_cluster_centers(self):
        return self.cluster_centers


class GeneticAlgo:
    def __init__(self, population_size, cities_coords, iteration, mutation_rate, initial_routes=None):
        self.population_size = population_size
        self.cities_coords = cities_coords
        self.iteration = iteration
        self.mutation_rate = mutation_rate
        self.initial_routes = initial_routes

        if initial_routes is None:
            self.population = self.generate_initial_solution()
        else:
            self.population = initial_routes
        self.best_path, self.best_distance = self.genetic_algorithm()

    def generate_initial_solution(self):
        num_cities = len(self.cities_coords)
        population = []
        for i in range(self.population_size):
            start_city = random.randint(0, num_cities-1)
            unvisited_cities = list(range(num_cities))
            unvisited_cities.remove(start_city)
            solution = [start_city]

            while unvisited_cities:
                last_city = solution[-1]
                distances = [distance(self.cities_coords[last_city], self.cities_coords[j])
                             for j in unvisited_cities]
                next_city = unvisited_cities[np.argmin(distances)]
                unvisited_cities.remove(next_city)
                solution.append(next_city)

            population.append(solution)

        return population

    # select
    def select(self, selection_rate=0.3):
        k = int(len(self.population) * selection_rate)
        participants = random.sample(self.population, k)
        winner = min(
            participants, key=lambda x: circuit_fitness(x, self.cities_coords))
        return winner

    # crossover
    def crossover(self, parent1, parent2):
        parent1_length = len(parent1)
        parent2_length = len(parent2)

        start = random.randint(0, len(parent1)-1)
        end = random.randint(start, len(parent1)-1)
        child = [-1 for i in range(parent1_length)]
        for i in range(start, end+1):
            child[i] = parent1[i]
        j = end+1
        for i in range(parent2_length):
            if parent2[i] not in child:
                if j == parent1_length:
                    j = 0
                child[j] = parent2[i]
                j += 1
        return child

    def mutate(self, path):
        if random.random() < self.mutation_rate:
            i = random.randint(0, len(path)-1)
            j = random.randint(0, len(path)-1)
            path[i], path[j] = path[j], path[i]
        return path

    def genetic_algorithm(self):
        global all_cities_coords
        best_path = []
        best_distance = float('inf')
        for i in range(self.iteration):
            print(i)
            new_population = []
            elite_size = int(self.population_size * 0)
            new_population.extend(
                sorted(self.population, key=lambda x: circuit_fitness(x, self.cities_coords))[:elite_size])

            for i in range(self.population_size - elite_size):
                parent1 = self.select()
                parent2 = self.select()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

            new_population = sorted(
                new_population, key=lambda x: circuit_fitness(x, self.cities_coords))
            new_best_path = new_population[0]
            new_best_distance = circuit_fitness(
                new_best_path, self.cities_coords)
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
                (all_cities_coords == self.cities_coords[city]).all(axis=1))[0][0]

        return best_path, best_distance

    def get_best_path(self):
        return self.best_path

    def get_best_distance(self):
        return self.best_distance


class AStar:
    def __init__(self, start, cities, all_cities_coords):
        self.start = start
        self.cities = cities
        self.all_cities_coords = all_cities_coords
        self.path, self.best_dist = self.a_star()

    def tsp_heuristic(self, state, remaining_cities):
        if not remaining_cities:
            return 0
        min_cost = math.inf
        for city in remaining_cities:
            cost = distance(
                self.all_cities_coords[state[-1]], self.all_cities_coords[city])
            new_state = state + [city]
            new_remaining_cities = [c for c in remaining_cities if c != city]
            cost += self.tsp_heuristic(new_state,
                                       new_remaining_cities)
            if cost < min_cost:
                min_cost = cost

        return min_cost

    def a_star(self):
        visited = [self.start]
        path = [self.start]
        remaining_cities = [c for c in self.cities if c not in visited]
        best_dist = 0

        while remaining_cities:
            nearest_city = None
            nearest_dist = float('inf')
            heuristic = None
            for i in remaining_cities:
                dist = distance(
                    self.all_cities_coords[path[-1]], self.all_cities_coords[i])
                heuristic = self.tsp_heuristic(
                    visited, remaining_cities)
                if dist + heuristic < nearest_dist:
                    nearest_city = i
                    nearest_dist = dist + heuristic
            visited.append(nearest_city)
            path.append(nearest_city)
            remaining_cities.remove(nearest_city)
            best_dist += nearest_dist - heuristic

        return path, best_dist

    def get_best_path(self):
        return self.path

    def get_best_dist(self):
        return self.best_dist


def main():

    # 클러스터 별 좌표 리스트와 클러스터 중심 좌표 리턴
    k = 7
    city_cluster = Cluster(all_cities_coords, k)

    cluster_coords = city_cluster.get_cluster_coords()
    cluster_centers = city_cluster.get_cluster_centers()

    # 클러스터 별 유전 알고리즘 돌리고 리스트에 저장
    mutation_rate = 0.3
    best_cluster = []
    for i in range(k):
        genetic_algo = GeneticAlgo(50, cluster_coords[i], 50, mutation_rate)
        best_path = genetic_algo.get_best_path()
        best_distance = genetic_algo.get_best_distance()
        best_cluster.append([best_path, best_distance])

    # 클러스터 첫, 마지막 도시 사이 좌표 저장
    for i in range(len(best_cluster)):
        first = best_cluster[i][0][0]
        last = best_cluster[i][0][-1]
        cluster_centers[i] = (all_cities_coords[first] +
                              all_cities_coords[last]) / 2

    # 클러스터 a star search 실행
    min_c_path = []
    min_c_distance = float('inf')
    for i in range(k):
        a_star = AStar(i, list(range(k)), cluster_centers)
        c_path = a_star.get_best_path()
        c_distance = a_star.get_best_dist()
        if c_distance < min_c_distance:
            min_c_distance = c_distance
            min_c_path = c_path

    # 순서에 따라 클러스터 이어붙이기
    #
    # 각각의 클러스터는 정방향 역방향 2가지 경우의 수
    # 2^10의 가지 총 1024가지 경로
    #
    all_reverse_cases = []
    n = len(min_c_path)
    for i in range(2**n):
        binary = bin(i)[2:].zfill(n)
        comb = []
        for j in range(n):
            if binary[j] == '0':
                comb.extend(best_cluster[min_c_path[j]][0])
            else:
                comb.extend(best_cluster[min_c_path[j]][0][::-1])
        all_reverse_cases.append(comb)

    best_distance = float('inf')
    best_path = []
    for i in all_reverse_cases:
        tmp = circuit_fitness(i, all_cities_coords)
        if tmp < best_distance:
            best_distance = tmp
            best_path = i

    res = best_distance

    #
    # 1024개의 케이스로 다시 유전 알고리즘 수행
    #
    new_population = []
    elite_size = 50
    new_population.extend(sorted(all_reverse_cases, key=lambda x: circuit_fitness(
        x, all_cities_coords))[:elite_size])
    genetic_algo2 = GeneticAlgo(
        elite_size, all_cities_coords, 1, mutation_rate, new_population)
    best_path = genetic_algo2.get_best_path()
    best_distance = genetic_algo2.get_best_distance()

    print("best distance : ", circuit_fitness(best_path, all_cities_coords))
    # 최적해 일부분(크기 5)만 최적 경로로 정렬 (1000번 반복)
    for i in range(1000):
        start_index = random.randint(0, len(best_path)-5)
        end_index = start_index + 5
        temp = best_path[start_index:end_index]

        coords = []
        for j in temp:
            coords.append(all_cities_coords[j])

        local_path = list(range(5))
        for k in range(5):
            a_star = AStar(k, list(range(5)), coords)
            path = a_star.get_best_path()
            if circuit_fitness(path, coords) < circuit_fitness(local_path, coords):
                local_path = path

        global_path = []
        for l in range(5):
            idx = local_path[l]
            global_path.append(temp[idx])

        new_best_path = best_path[:start_index] + \
            global_path[:] + best_path[end_index:]
        if circuit_fitness(best_path, all_cities_coords) > circuit_fitness(new_best_path, all_cities_coords):
            best_path = new_best_path

    for i in range(len(best_path)-1):
        plt.plot([all_cities_coords[best_path[i]][0], all_cities_coords[best_path[i+1]][0]],
                 [all_cities_coords[best_path[i]][1], all_cities_coords[best_path[i+1]][1]], color='k')
    plt.plot([all_cities_coords[best_path[-1]][0], all_cities_coords[best_path[0]][0]],
             [all_cities_coords[best_path[-1]][1], all_cities_coords[best_path[0]][1]], color='k')
    plt.show()

    idx = best_path.index(0)

    front = best_path[idx:]
    back = best_path[0:idx]

    best_path = front + back

    with open('./solution.csv', mode='w', newline='', encoding='utf-8-sig') as solution:
        writer = csv.writer(solution)
        for row in best_path:
            writer.writerow([row])
    print(len(set(best_path)))
    print("best distance : ", circuit_fitness(best_path, all_cities_coords))


if __name__ == "__main__":
    main()
