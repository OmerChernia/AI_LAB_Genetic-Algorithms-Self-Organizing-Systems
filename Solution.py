import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt
import math  # נדרש לחישוב האנטרופיה

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_TARGET = "Hello World!"
GA_CROSSOVER_OPERATOR = "SINGLE"  # Default; will be updated based on user input

# Global variable for fitness heuristic: "ORIGINAL" or "LCS"
GA_FITNESS_HEURISTIC = "ORIGINAL"  # Default; will be updated based on user input
GA_BONUS_FACTOR = 0.5  # Bonus factor for letters in the correct position

def lcs_length(s, t):
    """Compute the length of the Longest Common Subsequence between s and t."""
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]

class GAIndividual:  # Individual class for the GA
    def __init__(self, string=None):
        self.string = string if string else self.random_string()
        self.fitness = 0

    def random_string(self):  # Generates a random string of the same length as the target
        return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))

    def calculate_fitness(self):  # Original fitness: sum of absolute differences
        self.fitness = sum(abs(ord(self.string[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))

    # ---------- Task 7 ----------
    def calculate_fitness_lcs(self):
        """New fitness based on LCS with offset adjustment:
           fitness = (len(GA_TARGET) - LCS_length) - (GA_BONUS_FACTOR * number of exact matches)
                     + (GA_BONUS_FACTOR * len(GA_TARGET))
        """
        lcs = lcs_length(self.string, GA_TARGET)
        bonus = sum(1 for i in range(len(GA_TARGET)) if self.string[i] == GA_TARGET[i])
        offset = GA_BONUS_FACTOR * len(GA_TARGET)
        self.fitness = (len(GA_TARGET) - lcs) - (GA_BONUS_FACTOR * bonus) + offset

    def mutate(self):  # Mutates by changing a random character
        pos = random.randint(0, len(self.string) - 1)
        delta = chr((ord(self.string[pos]) + random.randint(0, 90)) % 122)
        s = list(self.string)
        s[pos] = delta
        self.string = ''.join(s)

def init_population():  # Initializes the population
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population):  # Sorts the population by fitness
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize):  # Copies the best esize individuals to the buffer
    buffer[:esize] = [GAIndividual(ind.string) for ind in population[:esize]]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness

# ---------- Task 4: Crossover Operators ----------
def crossover_single(parent1, parent2):
    tsize = len(parent1.string)
    spos = random.randint(0, tsize - 1)
    return parent1.string[:spos] + parent2.string[spos:]

def crossover_two(parent1, parent2):
    tsize = len(parent1.string)
    if tsize < 2:
        return crossover_single(parent1, parent2)
    point1 = random.randint(0, tsize - 2)
    point2 = random.randint(point1 + 1, tsize - 1)
    return parent1.string[:point1] + parent2.string[point1:point2] + parent1.string[point2:]

def crossover_uniform(parent1, parent2):
    tsize = len(parent1.string)
    child_chars = []
    for i in range(tsize):
        if random.random() < 0.5:
            child_chars.append(parent1.string[i])
        else:
            child_chars.append(parent2.string[i])
    return ''.join(child_chars)

def crossover_trivial(parent1, parent2):
    # Trivial crossover: returns one parent's string (randomly chosen)
    return parent1.string if random.random() < 0.5 else parent2.string

def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    tsize = len(GA_TARGET)
    elitism(population, buffer, esize)
    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2)
        i2 = random.randint(0, GA_POPSIZE // 2)
        # Select crossover operator based on GA_CROSSOVER_OPERATOR
        if GA_CROSSOVER_OPERATOR == "SINGLE":
            child_string = crossover_single(population[i1], population[i2])
        elif GA_CROSSOVER_OPERATOR == "TWO":
            child_string = crossover_two(population[i1], population[i2])
        elif GA_CROSSOVER_OPERATOR == "UNIFORM":
            child_string = crossover_uniform(population[i1], population[i2])
        elif GA_CROSSOVER_OPERATOR == "TRIVIAL":
            child_string = crossover_trivial(population[i1], population[i2])
        else:
            child_string = crossover_single(population[i1], population[i2])
        child = GAIndividual(child_string)
        if random.random() < GA_MUTATIONRATE:
            child.mutate()
        buffer.append(child)

# ---------- Task 9: Genetic Diversity Metrics (Factor Exploration) ----------
def compute_diversity_metrics(population):

    L = len(GA_TARGET)
    N = len(population)
    total_hamming = 0.0
    total_distinct = 0
    total_entropy = 0.0
    
    # For each gene position, compute frequencies
    for j in range(L):
        freq = {}
        for ind in population:
            allele = ind.string[j]
            freq[allele] = freq.get(allele, 0) + 1
        # (a) Average pairwise difference at this position: 1 - sum(p^2)
        pos_entropy_component = 0.0
        pos_p2_sum = 0.0
        for count in freq.values():
            p = count / N
            pos_p2_sum += p * p
            if p > 0:
                pos_entropy_component += -p * math.log2(p)
        avg_diff = 1 - pos_p2_sum  # probability two individuals differ at this gene
        total_hamming += avg_diff
        # (b) Number of distinct alleles at this position:
        total_distinct += len(freq)
        # (c) Entropy at this position:
        total_entropy += pos_entropy_component
    
    # Multiply avg_diff by L to get average Hamming distance per pair (over entire string)
    avg_hamming_distance = total_hamming * L
    avg_distinct = total_distinct / L  # average number of distinct alleles per position
    avg_entropy = total_entropy / L  # average entropy per position (in bits)
    
    return avg_hamming_distance, avg_distinct, avg_entropy

# ---------- Task 1: Generation Stats ----------
def print_generation_stats(population, generation, tick_duration, total_elapsed):
    fitness_values = [ind.fitness for ind in population]
    best = population[0]
    worst = population[-1]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    std_dev = statistics.stdev(fitness_values)
    fitness_range = worst.fitness - best.fitness
    print(f"Gen {generation}: Best = '{best.string}' (Fitness = {best.fitness})")
    print(f"  Avg Fitness = {avg_fitness:.2f}")
    print(f"  Std Dev = {std_dev:.2f}")
    print(f"  Worst Fitness = {worst.fitness}")
    print(f"  Fitness Range = {fitness_range}")
    print(f"  Tick Duration (sec) = {tick_duration:.4f}")
    print(f"  Total Elapsed Time (sec) = {total_elapsed:.4f}")
    
    # ---------- Task 8: Selection Pressure Metrics ----------
    adjusted = [worst.fitness - ind.fitness for ind in population]
    mean_adjusted = sum(adjusted) / len(adjusted)
    std_adjusted = statistics.stdev(adjusted)
    selection_variance = std_adjusted / mean_adjusted if mean_adjusted != 0 else 0
    total_adjusted = sum(adjusted)
    if total_adjusted == 0:
        probabilities = [1.0 / len(population)] * len(population)
    else:
        probabilities = [val / total_adjusted for val in adjusted]
    top_k = max(1, int(0.1 * len(population)))
    top_avg = sum(probabilities[:top_k]) / top_k
    overall_avg = 1.0 / len(population)
    top_avg_ratio = top_avg / overall_avg 
    print(f"  Selection Variance = {selection_variance:.6f}")
    print(f"  Top-Average Selection Probability Ratio = {top_avg_ratio:.2f}")
    
    # ---------- Task 9: Genetic Diversity Metrics ----------
    avg_hamming_distance, avg_distinct, avg_entropy = compute_diversity_metrics(population)
    print(f"  Avg Pairwise Hamming Distance = {avg_hamming_distance:.2f}")
    print(f"  Avg Number of Distinct Alleles per Gene = {avg_distinct:.2f}")
    print(f"  Avg Shannon Entropy per Gene (bits) = {avg_entropy:.2f}")
    print()

def main():
    # ---------- User Input for Fitness Heuristic ----------
    print("Select fitness heuristic:")
    print("1 - ORIGINAL (sum of differences)")
    print("2 - LCS-based")
    fitness_choice = input("Enter your choice (1/2): ")
    global GA_FITNESS_HEURISTIC
    if fitness_choice == "1":
        GA_FITNESS_HEURISTIC = "ORIGINAL"
    elif fitness_choice == "2":
        GA_FITNESS_HEURISTIC = "LCS"
    else:
        print("Invalid choice, defaulting to ORIGINAL")
        GA_FITNESS_HEURISTIC = "ORIGINAL"
    
    # ---------- User Input for Crossover Operator ----------
    print("Select crossover operator:")
    print("1 - SINGLE")
    print("2 - TWO")
    print("3 - UNIFORM")
    print("4 - TRIVIAL")
    choice = input("Enter your choice (1/2/3/4): ")
    global GA_CROSSOVER_OPERATOR
    if choice == "1":
        GA_CROSSOVER_OPERATOR = "SINGLE"
    elif choice == "2":
        GA_CROSSOVER_OPERATOR = "TWO"
    elif choice == "3":
        GA_CROSSOVER_OPERATOR = "UNIFORM"
    elif choice == "4":
        GA_CROSSOVER_OPERATOR = "TRIVIAL"
    else:
        print("Invalid choice, defaulting to SINGLE")
        GA_CROSSOVER_OPERATOR = "SINGLE"
    
    random.seed(time.time())
    start_time = timeit.default_timer()

    population = init_population()
    buffer = []
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []
    fitness_distributions = []

    for generation in range(GA_MAXITER):
        tick_start = timeit.default_timer()
        for ind in population:
            if GA_FITNESS_HEURISTIC == "ORIGINAL":
                ind.calculate_fitness()
            else:
                ind.calculate_fitness_lcs()
        sort_population(population)
        fitness_values = [ind.fitness for ind in population]
        fitness_distributions.append(fitness_values.copy())
        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)
        tick_end = timeit.default_timer()
        tick_duration = tick_end - tick_start
        total_elapsed = tick_end - start_time
        
        # ---------- Task 1 & Task 8 & Task 9: Generation Stats with Diversity Metrics ----------
        print_generation_stats(population, generation, tick_duration, total_elapsed)
        
        if population[0].fitness == 0:
            print(f"Converged after {generation + 1} generations.")
            break
        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population

    # ---------- Task 3_A: Fitness Behavior Plot ----------
    generations = list(range(len(best_fitness_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label="Best Fitness")
    plt.plot(generations, avg_fitness_list, label="Average Fitness")
    plt.plot(generations, worst_fitness_list, label="Worst Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Behavior per Generation")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------- Task 3_B: Box Plot of Fitness per Generation ----------
    plt.figure(figsize=(12, 6))
    plt.boxplot(fitness_distributions, showfliers=True)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Box Plot of Fitness per Generation')
    plt.grid(True)
    plt.show()
    
    # ---------- Task 5: Exploration vs. Exploitation Explanation ----------
    # The algorithm balances exploration and exploitation as follows:
    # • Exploration: Random initialization, mutation, and varied crossover operators introduce diversity
    #    and allow the search to explore new regions of the solution space.
    # • Exploitation: Sorting, elitism, and selecting parents from the top half ensure that the best solutions
    #    are propagated and refined over generations.

if __name__ == "__main__":
    main()
