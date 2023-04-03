import random
import numpy as np
import csv

cities = []
sol = []

#
# 유전 알고리즘만 사용
# ordered crossover
# 부모 두 개의 순서 유지, 일부 구간을 무작위로 선택 후 유전
# swap mutation
# 랜덤하게 선택된 두 위치의 값을 교환
#
# Partially Mapped Crossover(PMX)나 Order Crossover(OX)
# Swap Mutation나 Inversion Mutation


def distance(x, y) :
    dist = np.linalg.norm(np.array(x)-np.array(y))
    return dist


with open("./2023_AI_TSP.csv", mode='r', newline='', encoding='utf-8-sig') as tsp :
    reader = csv.reader(tsp)
    for row in reader :
        tmp = [float(row[0]), float(row[1])]
        cities.append(tmp)
total_cost = 0

num_cities = len(cities) 
gene_size = 100         
num_generations = 500
mutation_rate = 0.01

import random
import math

# 현재까지의 거리
def total_distance(city_order):
    total_distance = 0
    for i in range(len(city_order)):
        from_city = cities[city_order[i]]
        to_city = cities[city_order[(i+1) %len(city_order)]]
        total_distance += distance(from_city, to_city)
    return total_distance

# 초기 세팅 // 하나의 유전자는 도시 개수 만큼 // 랜덤으로 배치
def create_initial_gene():
    gene = []
    for i in range(gene_size):
        city_order = list(range(1, num_cities))
        random.shuffle(city_order)
        city_order.insert(0, 0)
        gene.append(city_order)
    return gene

# city_order에서 총 거리가 가장 짧은 애들로 2개 선택
def select(gene):
    fitness_list = []
    for i in range(len(gene)):
        city_order = gene[i]
        fitness = total_distance(city_order)
        fitness_list.append((city_order, fitness))
    fitness_list = sorted(fitness_list, key=lambda x:x[1])
    return fitness_list[0][0], fitness_list[1][0]

# 유전
def crossover(parent1, parent2):
    parent1_length = len(parent1)
    parent2_length = len(parent2)
    
    start = random.randint(1, len(parent1)-1)
    end = random.randint(start, len(parent1)-1)
    child = [-1 for i in range(parent1_length)]
    child[0] = 0
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

def mutation(city_order):
    if random.random() < mutation_rate:
        i = random.randint(0, len(city_order)-1)
        j = random.randint(0, len(city_order)-1)
        city_order[i], city_order[j] = city_order[j], city_order[i]
    return city_order

def genetic_algorithm():
    gene = create_initial_gene()
    best_distance = float('inf')
    best_city_order = None
    for i in range(num_generations):
        parent1, parent2 = select(gene)
        child = crossover(parent1, parent2)
        child = mutation(child)
        gene = [parent1, parent2, child]

        for j in range(gene_size-3):
            new_child = crossover(parent1, parent2)
            new_child = mutation(new_child)
            gene.append(new_child)

        fitness_list = []
        for city_order in gene:
            fitness = total_distance(city_order)
            fitness_list.append((city_order, fitness))
        fitness_list = sorted(fitness_list, key=lambda x:x[1])
        
        best_route = fitness_list[0]
        if best_route[1] < best_distance:
            best_distance = best_route[1]
            best_city_order = best_route[0]
            print("///////////////////////////////////////////////////////")
            print(best_distance) # 적합도 (총 거리)
            print(best_city_order[:10]) # 순서중 10개만 출력
            print("///////////////////////////////////////////////////////")

    return best_city_order, best_distance  



best_ordered, best_distance = genetic_algorithm()
 # print("Best ordered:", best_ordered)
print("Shortest distance:", best_distance)