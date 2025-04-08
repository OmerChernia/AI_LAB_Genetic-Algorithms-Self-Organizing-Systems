import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt
import math  # Required for entropy calculation
import json
import numpy as np

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_TARGET = "Hello World!"
GA_CROSSOVER_OPERATOR = "SINGLE"  # Default; will be updated based on user input

# Global variables for fitness heuristic and selection method:
GA_FITNESS_HEURISTIC = "ORIGINAL"  # or "LCS"
GA_BONUS_FACTOR = 0.5  # Bonus for correct position

# ---------- Task 10: Parent Selection Method Parameters ----------
# Options for parent's selection: "RWS", "SUS", "TournamentDet", "TournamentStoch", "Original"
GA_PARENT_SELECTION_METHOD = "RWS"  # Default; updated via user input
GA_TOURNAMENT_K = 5         # For tournament selection (deterministic or stochastic)
GA_TOURNAMENT_P = 0.8       # For non-deterministic tournament: probability to select the best
GA_MAX_AGE = 10             # Each individual lives for a fixed number of generations

# לאחר ההגדרות הקיימות
GA_MODE = "STRING"  # או "ARC"
GA_ARC_TARGET_GRID = None
GA_ARC_INPUT_GRID = None

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

class GAIndividual:
    def __init__(self, representation=None):
        self.repr = representation if representation else self.random_repr()
        self.fitness = 0
        self.age = 0

    def random_repr(self):
        if GA_MODE == "STRING":
            return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))
        else:  # ARC mode
            # Create random variation from input
            grid = [row.copy() for row in GA_ARC_INPUT_GRID]
            for _ in range(random.randint(1, 5)):  # Random number of initial mutations
                self.mutate_grid(grid)
            return grid

    def mutate_grid(self, grid):
        # Mutate grid - change random value
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        i = random.randint(0, rows-1)
        j = random.randint(0, cols-1)
        grid[i][j] = random.randint(0, 9)  # Assuming value range 0-9

    def calculate_fitness(self):
        if GA_MODE == "STRING":
            self.fitness = sum(abs(ord(self.repr[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))
        else:
            # Calculate match with target grid
            match_count = 0
            for i in range(len(GA_ARC_TARGET_GRID)):
                for j in range(len(GA_ARC_TARGET_GRID[0])):
                    if self.repr[i][j] == GA_ARC_TARGET_GRID[i][j]:
                        match_count += 1
            total_cells = len(GA_ARC_TARGET_GRID) * len(GA_ARC_TARGET_GRID[0])
            self.fitness = total_cells - match_count  # Lower value is better

    # ---------- Task 7 ----------
    def calculate_fitness_lcs(self):
        """New fitness based on LCS with offset adjustment."""
        if GA_MODE == "STRING":
            lcs = lcs_length(self.repr, GA_TARGET)
            bonus = sum(1 for i in range(len(GA_TARGET)) if self.repr[i] == GA_TARGET[i])
            offset = GA_BONUS_FACTOR * len(GA_TARGET)
            self.fitness = (len(GA_TARGET) - lcs) - (GA_BONUS_FACTOR * bonus) + offset

    def mutate(self):
        if GA_MODE == "STRING":
            pos = random.randint(0, len(self.repr) - 1)
            delta = chr((ord(self.repr[pos]) + random.randint(0, 90)) % 122)
            s = list(self.repr)
            s[pos] = delta
            self.repr = ''.join(s)

def init_population():
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population):
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize):
    buffer[:esize] = [GAIndividual(ind.repr) for ind in population[:esize]]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age

# ---------- Task 4: Crossover Operators ----------
def crossover_single(parent1, parent2):
    tsize = len(parent1.repr)
    spos = random.randint(0, tsize - 1)
    return parent1.repr[:spos] + parent2.repr[spos:]

def crossover_two(parent1, parent2):
    tsize = len(parent1.repr)
    if tsize < 2:
        return crossover_single(parent1, parent2)
    point1 = random.randint(0, tsize - 2)
    point2 = random.randint(point1 + 1, tsize - 1)
    return parent1.repr[:point1] + parent2.repr[point1:point2] + parent1.repr[point2:]

def crossover_uniform(parent1, parent2):
    tsize = len(parent1.repr)
    child_chars = []
    for i in range(tsize):
        child_chars.append(parent1.repr[i] if random.random() < 0.5 else parent2.repr[i])
    return ''.join(child_chars)

def crossover_trivial(parent1, parent2):
    return parent1.repr if random.random() < 0.5 else parent2.repr

def crossover_grid(parent1, parent2):
    grid1 = parent1.repr
    grid2 = parent2.repr
    child_grid = [row.copy() for row in grid1]
    
    # Horizontal or vertical crossover
    if random.random() < 0.5:
        # Horizontal split
        split_row = random.randint(1, len(grid1)-1)
        child_grid[split_row:] = grid2[split_row:]
    else:
        # Vertical split
        split_col = random.randint(1, len(grid1[0])-1)
        for i in range(len(grid1)):
            child_grid[i][split_col:] = grid2[i][split_col:]
    
    return child_grid

# ---------- Task 10: Parent Selection Methods ----------
def select_parent_RWS(population):
    worst = max(ind.fitness for ind in population)
    adjusted = [worst - ind.fitness for ind in population]
    total = sum(adjusted)
    if total == 0:
        return random.choice(population)
    r = random.uniform(0, total)
    cum = 0
    for ind, val in zip(population, adjusted):
        cum += val
        if cum >= r:
            return ind
    return population[-1]

def select_parent_TournamentDet(population):
    candidates = random.sample(population, GA_TOURNAMENT_K)
    return min(candidates, key=lambda ind: ind.fitness)

def select_parent_TournamentStoch(population):
    candidates = random.sample(population, GA_TOURNAMENT_K)
    candidates.sort(key=lambda ind: ind.fitness)
    for candidate in candidates:
        if random.random() < GA_TOURNAMENT_P:
            return candidate
    return candidates[-1]

def select_parents_SUS(population, num_parents):
    worst = max(ind.fitness for ind in population)
    adjusted = [worst - ind.fitness for ind in population]
    total = sum(adjusted)
    if total == 0:
        return [random.choice(population) for _ in range(num_parents)]
    step = total / num_parents
    start = random.uniform(0, step)
    pointers = [start + i * step for i in range(num_parents)]
    parents = []
    for p in pointers:
        cum = 0
        for ind, val in zip(population, adjusted):
            cum += val
            if cum >= p:
                parents.append(ind)
                break
    return parents

def select_parent_Original(population):
    # Original method: בוחרים באקראיות מתוך המחצית העליונה
    return random.choice(population[:len(population)//2])

# ---------- Task 10: Aging Survivor Selection ----------
def apply_aging(population):
    survivors = []
    for ind in population:
        ind.age += 1
        if ind.age < GA_MAX_AGE:
            survivors.append(ind)
    while len(survivors) < GA_POPSIZE:
        new_ind = GAIndividual()
        new_ind.age = 0
        survivors.append(new_ind)
    return survivors

# ---------- Task 9: Genetic Diversity Metrics (Factor Exploration) ----------
def compute_diversity_metrics(population):
    if GA_MODE == "STRING":
        L = len(GA_TARGET)
        N = len(population)
        total_hamming = 0.0
        total_distinct = 0
        total_entropy = 0.0
        for j in range(L):
            freq = {}
            for ind in population:
                allele = ind.repr[j]
                freq[allele] = freq.get(allele, 0) + 1
            pos_p2_sum = sum((count / N) ** 2 for count in freq.values())
            pos_entropy = -sum((count / N) * math.log2(count / N) for count in freq.values() if count > 0)
            avg_diff = 1 - pos_p2_sum
            total_hamming += avg_diff
            total_distinct += len(freq)
            total_entropy += pos_entropy
        avg_hamming_distance = total_hamming * L
        avg_distinct = total_distinct / L
        avg_entropy = total_entropy / L
        return avg_hamming_distance, avg_distinct, avg_entropy
    else:
        # Calculate diversity for matrices
        rows = len(GA_ARC_TARGET_GRID)
        cols = len(GA_ARC_TARGET_GRID[0])
        total_distinct = 0
        total_entropy = 0.0
        
        for i in range(rows):
            for j in range(cols):
                freq = {}
                for ind in population:
                    val = ind.repr[i][j]
                    freq[val] = freq.get(val, 0) + 1
                total_distinct += len(freq)
                for count in freq.values():
                    p = count / GA_POPSIZE
                    total_entropy -= p * math.log2(p) if p > 0 else 0
        
        avg_distinct = total_distinct / (rows * cols)
        avg_entropy = total_entropy / (rows * cols)
        return 0, avg_distinct, avg_entropy  # Hamming not relevant here

# ---------- Task 1: Generation Stats, Task 8 & Task 9 Combined ----------
def print_generation_stats(population, generation, tick_duration, total_elapsed):
    fitness_values = [ind.fitness for ind in population]
    best = population[0]
    worst = population[-1]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    std_dev = statistics.stdev(fitness_values)
    fitness_range = worst.fitness - best.fitness
    
    # Update display line for string mode
    if GA_MODE == "STRING":
        best_repr = f"'{best.repr}'"
    else:
        best_repr = f"Grid {best.repr[0][:3]}..."  # Truncated display for matrices
    
    print(f"Gen {generation}: Best = {best_repr} (Fitness = {best.fitness})")
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

# ---------- Task 10: Mating Function with Various Parent Selection Methods ----------
def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    num_offspring = GA_POPSIZE - esize
    sus_parents = []
    if GA_PARENT_SELECTION_METHOD == "SUS":
        sus_parents = select_parents_SUS(population, num_offspring * 2)
    for i in range(esize, GA_POPSIZE):
        if GA_PARENT_SELECTION_METHOD == "RWS":
            parent1 = select_parent_RWS(population)
            parent2 = select_parent_RWS(population)
        elif GA_PARENT_SELECTION_METHOD == "TournamentDet":
            parent1 = select_parent_TournamentDet(population)
            parent2 = select_parent_TournamentDet(population)
        elif GA_PARENT_SELECTION_METHOD == "TournamentStoch":
            parent1 = select_parent_TournamentStoch(population)
            parent2 = select_parent_TournamentStoch(population)
        elif GA_PARENT_SELECTION_METHOD == "SUS":
            parent1 = sus_parents.pop(0)
            parent2 = sus_parents.pop(0)
        elif GA_PARENT_SELECTION_METHOD == "Original":
            parent1 = select_parent_Original(population)
            parent2 = select_parent_Original(population)
        else:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
        if GA_MODE == "STRING":
            if GA_CROSSOVER_OPERATOR == "SINGLE":
                child_repr = crossover_single(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "TWO":
                child_repr = crossover_two(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "UNIFORM":
                child_repr = crossover_uniform(parent1, parent2)
            elif GA_CROSSOVER_OPERATOR == "TRIVIAL":
                child_repr = crossover_trivial(parent1, parent2)
        else:
            child_repr = crossover_grid(parent1, parent2)
        child = GAIndividual(child_repr)
        if random.random() < GA_MUTATIONRATE:
            if GA_MODE == "STRING":
                child.mutate()
            else:
                child.mutate_grid(child.repr)
        buffer.append(child)

def plot_grids(input_grid, target_grid, solution_grid=None):
    fig = plt.figure(figsize=(15, 5))
    
    # Input Grid
    ax1 = fig.add_subplot(131)
    ax1.matshow(input_grid, cmap='viridis')
    ax1.set_title("Input Grid")
    for (i, j), val in np.ndenumerate(input_grid):
        ax1.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    
    # Target Grid
    ax2 = fig.add_subplot(132)
    ax2.matshow(target_grid, cmap='viridis')
    ax2.set_title("Target Grid")
    for (i, j), val in np.ndenumerate(target_grid):
        ax2.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    
    # Solution Grid
    ax3 = fig.add_subplot(133)
    if solution_grid is not None:
        ax3.matshow(solution_grid, cmap='viridis')
        ax3.set_title("Solution Grid")
        for (i, j), val in np.ndenumerate(solution_grid):
            ax3.text(j, i, f'{val}', ha='center', va='center', color='w' if val > 5 else 'k')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No solution found', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def main():
    global GA_MODE, GA_ARC_TARGET_GRID, GA_ARC_INPUT_GRID
    
    print("Select mode:")
    print("1 - String evolution")
    print("2 - ARC puzzle")
    mode_choice = input("Enter your choice (1/2): ")
    
    if mode_choice == "2":
        GA_MODE = "ARC"
        json_path = input("Enter path to ARC JSON file: ")
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # הצגת רשימת התרגילים הקיימים
            num_examples = len(data['train'])
            print(f"\nFound {num_examples} training examples:")
            for i, example in enumerate(data['train']):
                input_grid = example['input']
                print(f"{i+1}. Input size: {len(input_grid)}x{len(input_grid[0])}")
            
            # בחירת תרגיל
            example_choice = int(input(f"\nSelect example (1-{num_examples}): ")) - 1
            if example_choice < 0 or example_choice >= num_examples:
                print("Invalid choice, using first example")
                example_choice = 0
                
            selected_example = data['train'][example_choice]
            GA_ARC_INPUT_GRID = selected_example['input']
            GA_ARC_TARGET_GRID = selected_example['output']
            
            # המרה ל-numpy arrays
            input_np = np.array(GA_ARC_INPUT_GRID)
            target_np = np.array(GA_ARC_TARGET_GRID)
            
            # בדיקת גודל מטריצות
            if len(GA_ARC_INPUT_GRID) != len(GA_ARC_TARGET_GRID) or \
               len(GA_ARC_INPUT_GRID[0]) != len(GA_ARC_TARGET_GRID[0]):
                raise ValueError("Input and target grids must have the same dimensions")
                
        except Exception as e:
            print(f"Error loading ARC puzzle: {e}")
            exit(1)
    else:
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
        print("5 - GRID")
        choice = input("Enter your choice (1/2/3/4/5): ")
        global GA_CROSSOVER_OPERATOR
        if choice == "1":
            GA_CROSSOVER_OPERATOR = "SINGLE"
        elif choice == "2":
            GA_CROSSOVER_OPERATOR = "TWO"
        elif choice == "3":
            GA_CROSSOVER_OPERATOR = "UNIFORM"
        elif choice == "4":
            GA_CROSSOVER_OPERATOR = "TRIVIAL"
        elif choice == "5":
            GA_CROSSOVER_OPERATOR = "GRID"
        else:
            print("Invalid choice, defaulting to SINGLE")
            GA_CROSSOVER_OPERATOR = "SINGLE"
        
        # ---------- Task 10: User Input for Parent Selection Method ----------
        print("Select parent selection method:")
        print("1 - RWS + Linear Scaling")
        print("2 - SUS + Linear Scaling")
        print("3 - Deterministic Tournament (K)")
        print("4 - Non-deterministic Tournament (P, K)")
        print("5 - Original (Random from top half)")
        sel_choice = input("Enter your choice (1/2/3/4/5): ")
        global GA_PARENT_SELECTION_METHOD
        if sel_choice == "1":
            GA_PARENT_SELECTION_METHOD = "RWS"
        elif sel_choice == "2":
            GA_PARENT_SELECTION_METHOD = "SUS"
        elif sel_choice == "3":
            GA_PARENT_SELECTION_METHOD = "TournamentDet"
        elif sel_choice == "4":
            GA_PARENT_SELECTION_METHOD = "TournamentStoch"
        elif sel_choice == "5":
            GA_PARENT_SELECTION_METHOD = "Original"
        else:
            print("Invalid choice, defaulting to RWS")
            GA_PARENT_SELECTION_METHOD = "RWS"
        
        try:
            k_val = int(input("Enter tournament parameter K (default 5): "))
            global GA_TOURNAMENT_K
            GA_TOURNAMENT_K = k_val
        except:
            GA_TOURNAMENT_K = 5
        try:
            p_val = float(input("Enter tournament probability P (default 0.8): "))
            global GA_TOURNAMENT_P
            GA_TOURNAMENT_P = p_val
        except:
            GA_TOURNAMENT_P = 0.8
        try:
            age_val = int(input("Enter maximum age (generations) for aging (default 10): "))
            global GA_MAX_AGE
            GA_MAX_AGE = age_val
        except:
            GA_MAX_AGE = 10

    random.seed(time.time())
    start_time = timeit.default_timer()

    population = init_population()
    buffer = []
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []
    fitness_distributions = []

    generation = 0
    best_solution = None
    while generation < GA_MAXITER:
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
        
        # ---------- Task 1, Task 8 & Task 9: Generation Stats with Diversity Metrics ----------
        print_generation_stats(population, generation, tick_duration, total_elapsed)
        
        if population[0].fitness == 0:
            best_solution = population[0].repr
            print(f"\n*** Converged after {generation + 1} generations ***")
            break
        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population
        # ---------- Task 10: Apply Aging ----------
        population = apply_aging(population)
        generation += 1

    if best_solution is None and GA_MODE == "ARC":
        print("\n*** Final Best Attempt ***")
        plot_grids(input_np, target_np, np.array(population[0].repr))
        print(f"Matching Cells: {len(GA_ARC_TARGET_GRID)*len(GA_ARC_TARGET_GRID[0]) - population[0].fitness}/{len(GA_ARC_TARGET_GRID)*len(GA_ARC_TARGET_GRID[0])}")
    elif GA_MODE == "ARC":
        solution_np = np.array(best_solution) if best_solution else None
        # הוספת ההצגה הגרפית לפני הגרפים
        plot_grids(input_np, target_np, solution_np)
        if best_solution is None:
            print(f"\nMatching Cells: {len(GA_ARC_TARGET_GRID)*len(GA_ARC_TARGET_GRID[0]) - population[0].fitness}/{len(GA_ARC_TARGET_GRID)*len(GA_ARC_TARGET_GRID[0])}")

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
    # • Exploitation: Sorting, elitism, and selecting parents based on the chosen selection method
    #    ensure that the best solutions are propagated and refined over generations.

if __name__ == "__main__":
    main()
