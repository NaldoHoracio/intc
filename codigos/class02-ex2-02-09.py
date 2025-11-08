# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 18:50:02 2025

@author: edvon
"""
import math
import random

class GA_Rastrigin:
    def __init__(self, pop_size=30, generations=100, mutation_rate=0.1):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(pop_size)]

    def fitness(self, ind):
        x, y = ind
        f = 20 + x**2 + y**2 - 10*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))
        return 1 / (1 + f)

    def selection(self):
        return random.choices(self.population, weights=[self.fitness(ind) for ind in self.population], k=2)

    def crossover(self, p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def mutate(self, ind):
        if random.random() < self.mutation_rate:
            return (ind[0] + random.uniform(-0.5, 0.5), ind[1] + random.uniform(-0.5, 0.5))
        return ind

    def run(self):
        for _ in range(self.generations):
            new_population = []
            for _ in range(self.pop_size):
                p1, p2 = self.selection()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

        best = min(self.population, key=lambda ind: 20 + ind[0]**2 + ind[1]**2 - 10*(math.cos(2*math.pi*ind[0]) + math.cos(2*math.pi*ind[1])))
        best_value = 20 + best[0]**2 + best[1]**2 - 10*(math.cos(2*math.pi*best[0]) + math.cos(2*math.pi*best[1]))
        return best, best_value

#%% Teste
ga3 = GA_Rastrigin()
print("Melhor solução encontrada:", ga3.run())