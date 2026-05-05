
from PIL import Image, ImageDraw
import numpy as np
import copy
import random
import time
import os

def draw(polygons, size):
    base = Image.new('RGBA', size, (255, 255, 255, 255))
    for poly in polygons:
        layer = Image.new('RGBA', size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(layer)
        draw.polygon(poly['vertices'], fill=poly['RGBA'])
        base = Image.alpha_composite(base, layer)
    return base.convert('RGB')

def random_triangle(size):
    w, h = size
    vertices = []
    for _ in range(3):
        x = random.randint(-20, w + 20)
        y = random.randint(-20, h + 20)
        vertices.append((x, y))
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = random.randint(30, 140)  # разумная прозрачность
    return {'vertices': vertices, 'RGBA': (r, g, b, a)}

def create_individual(n_triangles, size):
    return [random_triangle(size) for _ in range(n_triangles)]

def pixel_error(img1, img2):
    diff = np.array(img1) - np.array(img2)
    return np.sum(np.abs(diff))

def fitness(individual, target_img, size):
    rendered = draw(individual, size)
    error = pixel_error(rendered, target_img)
    max_possible_error = pixel_error(Image.new('RGB', size, (255,255,255)), target_img)
    return 100 * (1 - error / max_possible_error) if max_possible_error > 0 else 0.0

def mutate_triangle(triangle, size, mutation_rate=0.25):
    mutant = copy.deepcopy(triangle)
    w, h = size

    if random.random() < mutation_rate:
        # мутация вершин
        idx = random.randint(0, 2)
        dx = random.randint(-30, 30)
        dy = random.randint(-30, 30)
        x, y = mutant['vertices'][idx]
        mutant['vertices'][idx] = (x + dx, y + dy)

    if random.random() < mutation_rate:
        # мутация цвета / альфы
        c = list(mutant['RGBA'])
        idx = random.randint(0, 3)
        delta = random.randint(-40, 40)
        c[idx] = max(0, min(255, c[idx] + delta))
        mutant['RGBA'] = tuple(c)

    return mutant

def crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        return copy.deepcopy(parent1)  

    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def tournament_selection(population, fitnesses, k=4):
    selected = []
    for _ in range(len(population)):
        competitors = random.sample(list(zip(population, fitnesses)), k)
        winner = max(competitors, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def genetic_algorithm(target_img, n_tri=50, pop_size=30, max_gen=1500, elite_rate=0.15, mut_rate=0.3, outdir="output_gen"):
    size = target_img.size
    os.makedirs(outdir, exist_ok=True)

    # Инициализация популяции
    population = [create_individual(n_tri, size) for _ in range(pop_size)]
    fitnesses = [fitness(ind, target_img, size) for ind in population]

    print("Старт генетического алгоритма...")
    print(f"Generation: 0   Max Fitness: {max(fitnesses):6.2f}%   Avg: {np.mean(fitnesses):6.2f}%")

    best_fitness_history = []

    for gen in range(1, max_gen + 1):
        # Элитизм
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        num_elite = max(2, int(pop_size * elite_rate))
        new_pop = [copy.deepcopy(sorted_pop[i][0]) for i in range(num_elite)]

        # Отбор + кроссовер + мутация
        selected = tournament_selection(population, fitnesses)
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            # мутация каждого треугольника с вероятностью
            for i in range(len(child)):
                if random.random() < mut_rate:
                    child[i] = mutate_triangle(child[i], size)
            new_pop.append(child)

        # Оценка нового поколения
        new_fitnesses = [fitness(ind, target_img, size) for ind in new_pop]
        population = new_pop
        fitnesses = new_fitnesses

        max_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        best_fitness_history.append(max_fit)

        if gen % 100 == 0 or gen == max_gen:
            print(f"Generation: {gen:4d}   Max Fitness: {max_fit:6.2f}%   Avg: {avg_fit:6.2f}%")
            best_ind = population[np.argmax(fitnesses)]
            img = draw(best_ind, size)
            img.save(os.path.join(outdir, f"gen_{gen:05d}_fit_{max_fit:.2f}.png"))

    # Финальный лучший результат
    best_idx = np.argmax(fitnesses)
    final_img = draw(population[best_idx], size)
    final_img.save(os.path.join(outdir, "final.png"))

    print(f"\nЗавершено. Лучшая приспособленность: {max(fitnesses):.2f}%")
    print(f"Изображения сохранены в папке: {outdir}")

# Запуск
if __name__ == "__main__":
    target_path = "bug.png" 
    target = Image.open(target_path).convert("RGB")

    genetic_algorithm(
        target,
        n_tri=60,          
        pop_size=40,
        max_gen=2000,
        elite_rate=0.2,
        mut_rate=0.35,
        outdir="output_gen"
    )

