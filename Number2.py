import random

import numpy as np
import csv

# given cities
cities = []

populationSize = 60  # 염색체 개수
s = 1000  # 염색체 하나의 유전자 개수
mutationRate = 0.2  # 돌연 변이 확률
targetValue = 20000
bestValue = 50000


# Euclidean distance measuring function
def distance(x, y):
    dist = np.linalg.norm (np.array (x) - np.array (y))
    return dist


# 0. 후보해 집합 생성
class Chromosome:
    def __init__(self, g=[]):
        self.genes = g
        self.fitness = 0
        if self.genes.__len__ () == 0:
            temp_list = list (range (1, 1000))
            random.shuffle (temp_list)
            # ordering data set randomly
            self.genes = temp_list.copy ()

    def cal_fitness(self):

        self.fitness = 0
        value = 0

        for i in range (len (self.genes)):
            pos_city_1 = cities[self.genes[i]]
            pos_city_2 = cities[self.genes[(i + 1) % len (self.genes)]]
            dist = distance (pos_city_1, pos_city_2)

            # accumulation
            value += dist

        self.fitness = value
        return self.fitness


def print_p(pop):
    print (pop[0].genes[:10], "적합도=", pop[0].fitness)
    print ("")


# 선택 연산
def select(pop):
    max_value = sum ([c.cal_fitness () for c in pop])
    pick = random.uniform (0, max_value)
    # a <= x <= b의 실수 반환
    current = 0

    for c in pop:
        current += c.cal_fitness ()
        if current > pick:
            return c


# 교차 연산
def crossover(pop):
    father = select (pop)
    mother = select (pop)
    length = random.randint (2, s - 1)
    idx = random.randint (1, s - length)
    # a <= x <= b
    t_child1 = mother.genes[idx:idx + length].copy ()
    t_child2 = father.genes[idx:idx + length].copy ()

    child1 = list (filter (lambda x: not x in t_child1, father.genes))
    child2 = list (filter (lambda x: not x in t_child2, mother.genes))

    child1 = child1[:idx] + t_child1 + child1[idx:]
    child2 = child2[:idx] + t_child2 + child2[idx:]

    return child1, child2


# 돌연 변이 연산
def mutation(c):
    if random.random () < mutationRate:
        # 0.0 <= ~ < 1.0 실수를 반환
        x, y = random.sample (list (range (0, 999)), 2)
        c.genes[y], c.genes[x] = c.genes[x], c.genes[y]


# 1. get solution sequence and reordering (sort from 0)
with open ('TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    reader = csv.reader (tsp)
    for row in reader:
        tmp = [float (row[0]), float (row[1])]
        cities.append (tmp)

population = []
i = 0
fitness_list = []

while i < populationSize:
    population.append (Chromosome ())
    i += 1

count = 0
population.sort (key=lambda x: x.cal_fitness ())
count = 1

max_fitness = 0

print (population[0].genes[:10])
print (population[0].fitness)

while population[0].fitness > targetValue:
    if population[0].fitness > max_fitness:
        mutationRate = mutationRate * 0.9
        max_fitness = population[0].fitness
    new_pop = []

    # 선택과 교차 연산
    for _ in range (populationSize // 2):
        c1, c2 = crossover (population)
        new_pop.append (Chromosome (c1))
        new_pop.append (Chromosome (c2))

    population = new_pop.copy ()

    # 돌연 변이 연산
    for c in population:
        mutation (c)

    population.sort (key=lambda x: x.cal_fitness ())

    if population[0].fitness < bestValue:
        bestValue = population[0].fitness
    fitness_list.append (population[0].fitness)

    print("\nbestValue: ", bestValue)

    if population[0].fitness <= bestValue:
        print ("세대 번호= ", count)
        print_p (population)

    count += 1
    if count > 1000:
        break
