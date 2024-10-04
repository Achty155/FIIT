import numpy as np
import random
import matplotlib.pyplot as plt

# Vygenerovanie nahodnych miest v 200 x 200 priestore
def generate_cities(n_cities, map_size=200):
    return np.random.randint(0, map_size, size=(n_cities, 2))


#Výpočet Euklidovskej vzdialenosti medzi dvoma mestami
def Eukl_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Výpočet celkovej dĺžky trasy (permutácie miest)
def total_distance(cities, route):
    dist = 0
    for i in range(len(route)):
        dist += Eukl_distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return dist

#Inicializácia prvej generácie
def initialize_population(pop_size, n_cities):
    population = []
    for i in range(pop_size):
        individual = np.random.permutation(n_cities)
        population.append(individual)
    return population

#Fitness funkcia - prevrátená hodnota dĺžky trasy (aby sa minimalizovala dĺžka trasy)
def fitness(cities, individual):
    return 1 / total_distance(cities, individual)

# Výber rodičov pomocou rulety
def roulette_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=selection_probs)]

# Výber rodičov pomocou turnajového výberu
def tournament_selection(population, fitnesses, k = 3):
    k = min(len(population), k)
    selected_indices = np.random.choice(len(population), k, replace=False)

    best_index = selected_indices[0]
    for i in selected_indices:
        if fitnesses[i] > fitnesses[best_index]:
            best_index = i

    return population[best_index]


#Crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]

    pos = end
    for city in parent2:
        if city not in child:
            if pos >= size:
                pos = 0
            child[pos] = city
            pos += 1
    return child

# Mutácia - obrátenie úseku
def mutate(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end] = reversed(individual[start:end])
    return individual


# Evolúcia novej generácie s elitizmom
def evolve_population(population, pop_size, cities, mutation_rate):
    new_population = []
    best_indivs_size = 8
    fitnesses = [fitness(cities, individual) for individual in population]

    # Zistenie najlepších jedincov z predchodzej populácie
    best_indivs_indices = np.argsort(fitnesses)[-best_indivs_size:]  # Získanie indexov najlepších jedincov
    best_indivs = [population[i] for i in best_indivs_indices]

    # Pridanie  najlepších jedincov do novej populácie
    new_population.extend(best_indivs)

    # Zvyšok populácie generujeme pomocou výberu a crossoveru
    for i in range(pop_size - best_indivs_size):
        selection_method = random.choice([0, 1])
        if selection_method == 0:
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
        else:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

        child = crossover(parent1, parent2)

        if random.random() < mutation_rate:
            mutate(child)

        new_population.append(child)

    return new_population

# Vizuálne zobrazenie trasy
def plot_route(cities, route):
    plt.figure(figsize=(8, 8))
    route_cities = cities[route + [route[0]]]  # Uzavretie trasy
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'o-', markersize=10)
    plt.show()




# Hlavná funkcia
n_cities = 30
pop_size = 200
generations = 500
mutation_rate = 0.3

cities = generate_cities(n_cities)
population = initialize_population(pop_size, n_cities)

best_route = None
best_distance = float('inf')
best_distances = []

for gen in range(generations):
    population = evolve_population(population, pop_size, cities, mutation_rate)
    best_individual = min(population, key=lambda x: total_distance(cities, x))
    best_individual_distance = total_distance(cities, best_individual)

    if best_individual_distance < best_distance:
        best_route = best_individual
        best_distance = best_individual_distance

    best_distances.append(best_distance)




def convert_route_to_coordinates(cities, route):
    return [(int(cities[i][0]), int(cities[i][1])) for i in route]



best_route_coords = convert_route_to_coordinates(cities, best_route)
# Výsledky
print(f"Najkratšia nájdená trasa: {best_route_coords}")
print(f"Dĺžka trasy: {best_distance}")

# Zobrazenie trasy
plot_route(cities, best_route)

# Zobrazenie vývoja fitness (dĺžky trasy)
plt.plot(best_distances)
plt.xlabel("Generácia")
plt.ylabel("Dĺžka trasy")
plt.show()