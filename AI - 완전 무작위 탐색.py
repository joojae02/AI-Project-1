#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import random as r

cities = []
sol= []

# 유클리디언 거리 계산
def distance(a, b):
    dist = np.linalg.norm(np.array(a)-np.array(b))
    return dist


# In[154]:


# 좌표 값 csv 파일 받아와 cities에 저장
with open('2023_AI_TSP.csv', mode='r', newline='', encoding='utf-8-sig') as tsp:
    rd = csv.reader(tsp)
    for row in rd:
        cities.append(row)


# In[155]:


# 좌표 간 거리 계산
def t_cost (find_path, cities):
    total_cost = 0
    
    # 도시의 좌표 받아오기 (pos_city_2가 (idx+1이므로 len(path)-1))
    for idx in range(len(find_path)-1):
        pos_city_1 = [float(cities[find_path[idx]][0]), float(cities[find_path[idx]][1])]
        pos_city_2 = [float(cities[find_path[idx+1]][0]), float(cities[find_path[idx+1]][1])]
        
        # 도시 간 거리 계산
        dist = distance(pos_city_1, pos_city_2)
        
        total_cost += dist
        
    return total_cost


# In[156]:


# 완전 무작위 서치
def search (cities, N):
    best_cost = float("inf")
    sol_path = None
    
    # cities 길이만큼 list 받아와 shuffle로 리스트 순서 무작위로 섞기
    for i in range(N):
        find_path = list(range(len(cities)))
        r.shuffle(path)
        # --> or.. time 통해 일정 시간 설정해 choice로?
        
        # 순서 재배치
        idx = find_path.index(0)
        
        front = find_path[idx:]
        back = find_path[0:idx]
        
        find_path = front + back
        
        # 시작점으로 돌아가기
        find_path.append(int(0)) 
        
        # 좌표 간 거리 계산해 cost에 저장
        cost = t_cost(find_path, cities)
        
        # cost가 bset_cost보다 작으면 cost 값 best_cost에 저장, path 값 sol_path에 저장
        if cost < best_cost:
            best_cost = cost
            sol_path = find_path
            
    return best_cost, sol_path

best_cost, sol = search (cities, len(cities))

# solution csv 파일 만들어 sol 값 입력
with open('solution.csv', mode='w', newline='', encoding='utf-8-sig') as solution:
    wt = csv.writer(solution)
    for row in sol:
        wt.writerow([row])

# Best Cost 출력
print('Best Cost of Search : ' + str(best_cost))


# In[ ]:




