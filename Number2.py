import random

import numpy as np
import csv

# given cities
cities = []
# solution
sol = []


populationSize = 60
s = 1000
mutationRate = 0.5
targetValue = 40000
bestValue = 40000

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
            temp_list = list (range (1000))
            random.shuffle (temp_list)
            # ordering data set randomly
            self.genes = temp_list.copy ()

    def cal_fitness(self):

        self.fitness = 0
        value = 0
        global bestValue

        for i in range (len(cities)-1):
            pos_city_1 = [float (cities[sol[idx]][0]), float (cities[sol[idx]][1])]
            pos_city_2 = [float (cities[sol[idx + 1]][0]), float (cities[sol[idx + 1]][1])]
            # distance calculation
            dist = distance (pos_city_1, pos_city_2)

            # accumulation
            value += dist

        self.fitness = ((targetValue - value) + (targetValue - bestValue)) / 3
        if self.fitness < bestValue:
            bestValue = self.fitness
        return self.fitness

def print_p(pop):
    i = 0
    for x in pop:
        print("염색체 #", i, "=", x, "적합도=", x.fitness)
        i += 1
    print("")


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
        x, y = random.sample (list (range (1, 1000)), 2)
        c.genes[y], c.genes[x] = c.genes[x], c.genes[y]


# 1. get solution sequence and reordering (sort from 0)
with open ('example_solution.csv', mode='r', newline='', encoding='utf-8-sig') as solution:
    # read solution sequence
    reader = csv.reader (solution)
    for row in reader:
        sol.append (int (row[0]))

    # reordering solution sequence
    idx = sol.index (0)
    front = sol[idx:]
    back = sol[0:idx]

    sol = front + back

    # expand 0 city (start) for simplicity
    sol.append (int (0))

# 2. get TSP city map
with open ('TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    # read TSP city map
    reader = csv.reader (tsp)
    for row in reader:
        cities.append (row)

# 3. evaluate solution cost
total_cost = 0

for idx in range (len (sol) - 1):
    # get city positions
    pos_city_1 = [float (cities[sol[idx]][0]), float (cities[sol[idx]][1])]
    pos_city_2 = [float (cities[sol[idx + 1]][0]), float (cities[sol[idx + 1]][1])]

    # distance calculation
    dist = distance (pos_city_1, pos_city_2)

    # accumulation
    total_cost += dist

print ('final cost: ' + str (total_cost))

population = []
i = 0
fitness_list = []

while i < populationSize:
    population.append (Chromosome ())
    i += 1

count = 0
population.sort (key=lambda x: x.cal_fitness (), reverse=True)
print ("generation number: ", count)
count = 1

max_fitness = targetValue

print(population[0].genes)
print(population[0].fitness)

while population[0].fitness < targetValue:
    if population[0].fitness < max_fitness:
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

    population.sort (key=lambda x: x.cal_fitness (), reverse=True)
    fitness_list.append (population[0].fitness)
    print ("세대 번호=", count)
    print_p (population)
    count += 1
    if count > 50000: break