import random
import time
import timeit
import statistics
import math  # נדרש לחישוב אנטרופיה
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- GA Parameters -----------
GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_CROSSOVER_OPERATOR = "SINGLE"  # Default; will be updated based on user input

# Global variables for fitness heuristic and selection method:
GA_FITNESS_HEURISTIC = "ORIGINAL"  # "ORIGINAL", "LCS", or "BINPACK"
GA_BONUS_FACTOR = 0.5  # Bonus factor for LCS
GA_PARENT_SELECTION_METHOD = "RWS"
GA_TOURNAMENT_K = 5
GA_TOURNAMENT_P = 0.8
GA_MAX_AGE = 10

# ---------- Task 1: Hello World -----------
GA_TARGET = "Hello World!"

# ---------- Task 11: BINPACK Globals (Problem Mode)
BINPACK_CAPACITY = 150
BINPACK_NUM_ITEMS = 0
BINPACK_BEST_KNOWN = 9999
BINPACK_VOLUMES = []
BINPACK_MAX_BINS = 0  # יוגדר לפי BINPACK_NUM_ITEMS

###############################################################################
#                               GAIndividual Class
###############################################################################
class GAIndividual:
    """
    מחלקה אחת שמסוגלת לייצג:
    1) שרשרת תווים (HELLO WORLD / LCS)
    2) רשימת הקצאות למיכלים (Bin Packing)
    בהתאם ל־GA_FITNESS_HEURISTIC
    """
    def __init__(self, genotype=None):
        self.fitness = 0
        self.age = 0  # ---------- Task 10: Aging attribute

        # נבדוק במצב ריצה האם אנחנו ב-"ORIGINAL"/"LCS" או "BINPACK"
        if GA_FITNESS_HEURISTIC in ["ORIGINAL", "LCS"]:
            if genotype is not None:
                self.genotype = genotype  # מחרוזת
            else:
                self.genotype = self.random_string_hello()
        elif GA_FITNESS_HEURISTIC == "BINPACK":
            if genotype is not None:
                self.genotype = genotype  # list של אינדקסים
            else:
                self.genotype = self.random_solution_binpack()
        else:
            # ברירת מחדל
            if genotype is not None:
                self.genotype = genotype
            else:
                self.genotype = self.random_string_hello()

    # ---------- Task 1: random_string (Hello World) ----------
    def random_string_hello(self):
        return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))

    # ---------- Task 11: random_solution (Bin Packing) ----------
    def random_solution_binpack(self):
        # כל פריט מקבל binIndex בין 0..BINPACK_MAX_BINS-1
        return [random.randint(0, BINPACK_MAX_BINS-1) for _ in range(BINPACK_NUM_ITEMS)]

    # ---------- Original fitness (sum of absolute differences) ----------
    def calculate_fitness_original(self):
        # sum of absolute differences from GA_TARGET
        s = self.genotype
        self.fitness = sum(abs(ord(s[i]) - ord(GA_TARGET[i])) 
                           for i in range(len(GA_TARGET)))

    # ---------- Task 7: LCS-based fitness ----------
    def calculate_fitness_lcs(self):
        s = self.genotype
        lcs_len = self.lcs_length(s, GA_TARGET)
        bonus = sum(1 for i in range(len(GA_TARGET)) if s[i] == GA_TARGET[i])
        offset = GA_BONUS_FACTOR * len(GA_TARGET)
        self.fitness = (len(GA_TARGET) - lcs_len) - (GA_BONUS_FACTOR * bonus) + offset

    @staticmethod
    def lcs_length(s, t):
        m, n = len(s), len(t)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[m][n]

    # ---------- Bin Packing fitness ----------
    def calculate_fitness_binpack(self):
        bin_usage = defaultdict(int)
        for item_idx, bin_idx in enumerate(self.genotype):
            bin_usage[bin_idx] += BINPACK_VOLUMES[item_idx]
        used_bins = 0
        penalty = 0
        alpha = 1000
        for b_idx, total_vol in bin_usage.items():
            if total_vol > 0:
                used_bins += 1
                if total_vol > BINPACK_CAPACITY:
                    penalty += alpha*(total_vol - BINPACK_CAPACITY)
        self.fitness = used_bins + penalty

    # ---------- Decide which fitness to calculate ----------
    def calculate_fitness(self):
        if GA_FITNESS_HEURISTIC == "ORIGINAL":
            self.calculate_fitness_original()
        elif GA_FITNESS_HEURISTIC == "LCS":
            self.calculate_fitness_lcs()
        elif GA_FITNESS_HEURISTIC == "BINPACK":
            self.calculate_fitness_binpack()
        else:
            self.calculate_fitness_original()

    # ---------- mutate (Hello World) ----------
    def mutate_hello(self):
        s = list(self.genotype)
        pos = random.randint(0, len(s)-1)
        delta = chr((ord(s[pos]) + random.randint(0, 90)) % 122)
        s[pos] = delta
        self.genotype = ''.join(s)

    # ---------- mutate (Bin Packing) ----------
    def mutate_binpack(self):
        if BINPACK_NUM_ITEMS > 0:
            pos = random.randint(0, BINPACK_NUM_ITEMS - 1)
            self.genotype[pos] = random.randint(0, BINPACK_MAX_BINS-1)

    def mutate(self):
        if GA_FITNESS_HEURISTIC in ["ORIGINAL", "LCS"]:
            self.mutate_hello()
        elif GA_FITNESS_HEURISTIC == "BINPACK":
            self.mutate_binpack()
        else:
            self.mutate_hello()

###############################################################################
#                                  GA CORE
###############################################################################
def init_population():
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population):
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize):
    buffer[:esize] = [GAIndividual(ind.genotype) for ind in population[:esize]]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age

# ---------- Task 4: Crossover Operators ----------
def crossover_single(parent1, parent2):
    length = len(parent1.genotype)
    spos = random.randint(0, length-1)
    # אם String => מחבר מחרוזות, אם List => מחבר רשימות
    if isinstance(parent1.genotype, str):
        return parent1.genotype[:spos] + parent2.genotype[spos:]
    else:
        return parent1.genotype[:spos] + parent2.genotype[spos:]

def crossover_two(parent1, parent2):
    length = len(parent1.genotype)
    if length < 2:
        return crossover_single(parent1, parent2)
    p1 = random.randint(0, length-2)
    p2 = random.randint(p1+1, length-1)
    if isinstance(parent1.genotype, str):
        return (parent1.genotype[:p1] 
                + parent2.genotype[p1:p2] 
                + parent1.genotype[p2:])
    else:
        return (parent1.genotype[:p1] 
                + parent2.genotype[p1:p2] 
                + parent1.genotype[p2:])

def crossover_uniform(parent1, parent2):
    length = len(parent1.genotype)
    new_sol = []
    for i in range(length):
        if random.random() < 0.5:
            new_sol.append(parent1.genotype[i])
        else:
            new_sol.append(parent2.genotype[i])
    if isinstance(parent1.genotype, str):
        return ''.join(new_sol)
    else:
        return new_sol

def crossover_trivial(parent1, parent2):
    return parent1.genotype if random.random()<0.5 else parent2.genotype

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
    return random.choice(population[:len(population)//2])

def apply_aging(population):
    survivors = []
    for ind in population:
        ind.age += 1
        if ind.age < GA_MAX_AGE:
            survivors.append(ind)
    while len(survivors) < GA_POPSIZE:
        survivors.append(GAIndividual())
    return survivors

# ---------- Task 9: Genetic Diversity Metrics (Factor Exploration) ----------
def compute_diversity_metrics(population):
    if GA_FITNESS_HEURISTIC in ["ORIGINAL","LCS"]:
        length = len(GA_TARGET)
    elif GA_FITNESS_HEURISTIC == "BINPACK":
        length = BINPACK_NUM_ITEMS
    else:
        length = len(GA_TARGET)
    N = len(population)
    total_hamming = 0.0
    total_distinct = 0
    total_entropy = 0.0

    for j in range(length):
        freq = {}
        for ind in population:
            allele = ind.genotype[j]
            freq[allele] = freq.get(allele, 0) + 1

        pos_p2_sum = sum((count / N)**2 for count in freq.values())
        pos_entropy = -sum((count/N)*math.log2(count/N) for count in freq.values() if count>0)
        avg_diff = 1 - pos_p2_sum
        total_hamming += avg_diff
        total_distinct += len(freq)
        total_entropy += pos_entropy

    avg_hamming_distance = total_hamming * length
    avg_distinct = total_distinct / length
    avg_entropy = total_entropy / length
    return avg_hamming_distance, avg_distinct, avg_entropy

# ---------- Task 1: Generation Stats & Task 8 & Task 9 Combined ----------
def print_generation_stats(population, generation, tick_duration, total_elapsed):
    fitness_values = [ind.fitness for ind in population]
    best = population[0]
    worst = population[-1]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    std_dev = statistics.stdev(fitness_values)
    fitness_range = worst.fitness - best.fitness

    # הדפסה נוחה
    if isinstance(best.genotype, str):
        best_repr = f"'{best.genotype}'"
    else:
        # מציגים רק התחלה
        best_repr = f"{best.genotype[:10]}... (len={len(best.genotype)})"

    print(f"Gen {generation}: Best = {best_repr} (Fitness = {best.fitness})")
    print(f"  Avg Fitness = {avg_fitness:.2f}")
    print(f"  Std Dev = {std_dev:.2f}")
    print(f"  Worst Fitness = {worst.fitness}")
    print(f"  Fitness Range = {fitness_range}")
    print(f"  Tick Duration (sec) = {tick_duration:.4f}")
    print(f"  Total Elapsed Time (sec) = {total_elapsed:.4f}")

    # ---------- Task 8: Selection Pressure ----------
    adjusted = [worst.fitness - ind.fitness for ind in population]
    mean_adjusted = sum(adjusted)/len(adjusted)
    std_adjusted = statistics.stdev(adjusted)
    selection_variance = std_adjusted / mean_adjusted if mean_adjusted!=0 else 0
    total_adjusted = sum(adjusted)
    if total_adjusted == 0:
        probabilities = [1.0/len(population)]*len(population)
    else:
        probabilities = [val/total_adjusted for val in adjusted]
    top_k = max(1, int(0.1*len(population)))
    top_avg = sum(probabilities[:top_k]) / top_k
    overall_avg = 1.0/len(population)
    top_avg_ratio = top_avg/overall_avg
    print(f"  Selection Variance = {selection_variance:.6f}")
    print(f"  Top-Average Selection Probability Ratio = {top_avg_ratio:.2f}")

    # ---------- Task 9: Diversity ----------
    avg_hamming_distance, avg_distinct, avg_entropy = compute_diversity_metrics(population)
    print(f"  Avg Pairwise Hamming Distance = {avg_hamming_distance:.2f}")
    print(f"  Avg Number of Distinct Alleles per Gene = {avg_distinct:.2f}")
    print(f"  Avg Shannon Entropy per Gene (bits) = {avg_entropy:.2f}")
    print()

# ---------- Task 10: Mating (various parent selection) ----------
def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    num_offspring = GA_POPSIZE - esize
    sus_parents = []
    if GA_PARENT_SELECTION_METHOD == "SUS":
        sus_parents = select_parents_SUS(population, num_offspring*2)

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

        # crossover
        if GA_CROSSOVER_OPERATOR == "SINGLE":
            child_geno = crossover_single(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "TWO":
            child_geno = crossover_two(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "UNIFORM":
            child_geno = crossover_uniform(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "TRIVIAL":
            child_geno = crossover_trivial(parent1, parent2)
        else:
            child_geno = crossover_single(parent1, parent2)

        child = GAIndividual(child_geno)
        if random.random() < GA_MUTATIONRATE:
            child.mutate()
        buffer.append(child)

###############################################################################
#                               BINPACK Parsing
###############################################################################
def parse_binpack(file_path):
    """
    קובץ בפורמט:
    1) שורה ראשונה: מספר מופעים P
    לכל מופע:
      problem_id
      binCapacity, numItems, bestKnownBins
      ואח"כ numItems נפחים
    """
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    idx = 0
    P = int(lines[idx])
    idx += 1
    problems = []
    for _ in range(P):
        problem_id = lines[idx]
        idx += 1
        parts = lines[idx].split()
        idx += 1
        capacity = int(parts[0])
        nitems = int(parts[1])
        best_known = int(parts[2])
        volumes = []
        for __ in range(nitems):
            vol = int(lines[idx])
            idx += 1
            volumes.append(vol)
        problems.append((problem_id, capacity, nitems, best_known, volumes))
    return problems

###############################################################################
#                             MAIN
###############################################################################
def main():
    print("Select problem mode:")
    print("1 - HELLO WORLD + (Original / LCS)")
    print("2 - BINPACKING")
    choice = input("Enter your choice (1/2): ").strip()
    if choice == "2":
        run_binpacking_mode()
    else:
        run_hello_world_mode()

def run_hello_world_mode():
    global GA_FITNESS_HEURISTIC
    print("Select fitness heuristic:")
    print("1 - ORIGINAL (sum of differences)")
    print("2 - LCS-based")
    h_choice = input("Enter your choice (1/2): ").strip()
    if h_choice == "2":
        GA_FITNESS_HEURISTIC = "LCS"
    else:
        GA_FITNESS_HEURISTIC = "ORIGINAL"

    # ---------- User Input for Crossover Operator ----------
    global GA_CROSSOVER_OPERATOR
    print("Select crossover operator:")
    print("1 - SINGLE")
    print("2 - TWO")
    print("3 - UNIFORM")
    print("4 - TRIVIAL")
    c_choice = input("Enter your choice (1/2/3/4): ")
    if c_choice == "1":
        GA_CROSSOVER_OPERATOR = "SINGLE"
    elif c_choice == "2":
        GA_CROSSOVER_OPERATOR = "TWO"
    elif c_choice == "3":
        GA_CROSSOVER_OPERATOR = "UNIFORM"
    elif c_choice == "4":
        GA_CROSSOVER_OPERATOR = "TRIVIAL"
    else:
        GA_CROSSOVER_OPERATOR = "SINGLE"

    # ---------- Task 10: Parent Selection ----------
    global GA_PARENT_SELECTION_METHOD, GA_TOURNAMENT_K, GA_TOURNAMENT_P, GA_MAX_AGE
    print("Select parent selection method:")
    print("1 - RWS + Linear Scaling")
    print("2 - SUS + Linear Scaling")
    print("3 - Deterministic Tournament (K)")
    print("4 - Non-deterministic Tournament (P, K)")
    print("5 - Original (Random from top half)")
    sel_choice = input("Enter your choice (1/2/3/4/5): ")
    if sel_choice == "2":
        GA_PARENT_SELECTION_METHOD = "SUS"
    elif sel_choice == "3":
        GA_PARENT_SELECTION_METHOD = "TournamentDet"
    elif sel_choice == "4":
        GA_PARENT_SELECTION_METHOD = "TournamentStoch"
    elif sel_choice == "5":
        GA_PARENT_SELECTION_METHOD = "Original"
    else:
        GA_PARENT_SELECTION_METHOD = "RWS"

    try:
        k_val = int(input("Enter tournament parameter K (default 5): "))
        GA_TOURNAMENT_K = k_val
    except:
        GA_TOURNAMENT_K = 5
    try:
        p_val = float(input("Enter tournament probability P (default 0.8): "))
        GA_TOURNAMENT_P = p_val
    except:
        GA_TOURNAMENT_P = 0.8
    try:
        age_val = int(input("Enter maximum age (default 10): "))
        GA_MAX_AGE = age_val
    except:
        GA_MAX_AGE = 10

    # נריץ GA (Hello World / LCS)
    run_GA_hello_lcs()


def run_binpacking_mode():
    global GA_FITNESS_HEURISTIC
    GA_FITNESS_HEURISTIC = "BINPACK"
    binpack_file = input("Enter binpack file path (e.g. binpack1.txt): ").strip()
    problems = parse_binpack(binpack_file)
    print(f"Found {len(problems)} problems in {binpack_file}:")
    for i,(pid,cap,nitems,bestK,vols) in enumerate(problems):
        print(f"[{i}] ID={pid} capacity={cap} items={nitems} bestKnown={bestK}")
    choice = int(input("Which problem index to run? "))
    # נטמיע ערכים גלובליים
    pid, capacity, nitems, bestKnown, volumes = problems[choice]
    global BINPACK_CAPACITY, BINPACK_NUM_ITEMS, BINPACK_BEST_KNOWN, BINPACK_VOLUMES, BINPACK_MAX_BINS
    BINPACK_CAPACITY = capacity
    BINPACK_NUM_ITEMS = nitems
    BINPACK_BEST_KNOWN = bestKnown
    BINPACK_VOLUMES = volumes
    BINPACK_MAX_BINS = nitems  # הגבלה פשוטה

    # בחירת crossover
    global GA_CROSSOVER_OPERATOR
    print("Select crossover operator:")
    print("1 - SINGLE")
    print("2 - TWO")
    print("3 - UNIFORM")
    print("4 - TRIVIAL")
    c_choice = input("Enter your choice (1/2/3/4): ")
    if c_choice == "2":
        GA_CROSSOVER_OPERATOR = "TWO"
    elif c_choice == "3":
        GA_CROSSOVER_OPERATOR = "UNIFORM"
    elif c_choice == "4":
        GA_CROSSOVER_OPERATOR = "TRIVIAL"
    else:
        GA_CROSSOVER_OPERATOR = "SINGLE"

    # בחירת שיטת בחירה
    global GA_PARENT_SELECTION_METHOD, GA_TOURNAMENT_K, GA_TOURNAMENT_P, GA_MAX_AGE
    print("Select parent selection method:")
    print("1 - RWS + Linear Scaling")
    print("2 - SUS + Linear Scaling")
    print("3 - Deterministic Tournament (K)")
    print("4 - Non-deterministic Tournament (P, K)")
    print("5 - Original (Random from top half)")
    sel_choice = input("Enter your choice (1/2/3/4/5): ")
    if sel_choice == "2":
        GA_PARENT_SELECTION_METHOD = "SUS"
    elif sel_choice == "3":
        GA_PARENT_SELECTION_METHOD = "TournamentDet"
    elif sel_choice == "4":
        GA_PARENT_SELECTION_METHOD = "TournamentStoch"
    elif sel_choice == "5":
        GA_PARENT_SELECTION_METHOD = "Original"
    else:
        GA_PARENT_SELECTION_METHOD = "RWS"

    try:
        k_val = int(input("Enter tournament parameter K (default 5): "))
        GA_TOURNAMENT_K = k_val
    except:
        GA_TOURNAMENT_K = 5
    try:
        p_val = float(input("Enter tournament probability P (default 0.8): "))
        GA_TOURNAMENT_P = p_val
    except:
        GA_TOURNAMENT_P = 0.8
    try:
        age_val = int(input("Enter maximum age (default 10): "))
        GA_MAX_AGE = age_val
    except:
        GA_MAX_AGE = 10

    run_GA_binpack()

def run_GA_hello_lcs():
    random.seed(time.time())
    population = init_population()
    buffer = []
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []
    fitness_distributions = []

    start_time = timeit.default_timer()
    generation = 0
    while generation < GA_MAXITER:
        tick_start = timeit.default_timer()
        for ind in population:
            ind.calculate_fitness()
        sort_population(population)
        fitness_vals = [ind.fitness for ind in population]
        fitness_distributions.append(fitness_vals.copy())
        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(fitness_vals)/len(population)
        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)
        tick_end = timeit.default_timer()
        tick_duration = tick_end - tick_start
        total_elapsed = tick_end - start_time

        print_generation_stats(population, generation, tick_duration, total_elapsed)

        if best_fitness == 0:
            print(f"Converged after {generation+1} generations.")
            break

        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population
        population = apply_aging(population)
        generation += 1

    # ---------- Task 3_A: Fitness Behavior Plot ----------
    plot_results(best_fitness_list, avg_fitness_list, worst_fitness_list, fitness_distributions)

    # ---------- Task 5: Exploration vs. Exploitation Explanation ----------
    # The algorithm balances exploration and exploitation as follows:
    # • Exploration: Random initialization, mutation, and varied crossover operators introduce diversity
    #    and allow the search to explore new regions of the solution space.
    # • Exploitation: Sorting, elitism, and selecting parents from the top half ensure that the best solutions
    #    are propagated and refined over generations.

def run_GA_binpack():
    random.seed(time.time())
    population = init_population()
    buffer = []
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []
    fitness_distributions = []

    start_time = timeit.default_timer()
    generation = 0
    while generation < GA_MAXITER:
        tick_start = timeit.default_timer()
        for ind in population:
            ind.calculate_fitness()
        sort_population(population)
        fitness_vals = [ind.fitness for ind in population]
        fitness_distributions.append(fitness_vals.copy())
        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(fitness_vals)/len(population)
        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)
        tick_end = timeit.default_timer()
        tick_duration = tick_end - tick_start
        total_elapsed = tick_end - start_time

        print_generation_stats(population, generation, tick_duration, total_elapsed)

        # אם best_fitness==0 => כנראה שימוש במיכל אחד וללא חריגה => אפשר לעצור
        if best_fitness == 0:
            print(f"Converged after {generation+1} generations.")
            break

        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population
        population = apply_aging(population)
        generation += 1

    # סיום
    #decode_binpack_solution(population[0])

    # ---------- Task 3_B: Box Plot of Fitness per Generation ----------
    plot_results(best_fitness_list, avg_fitness_list, worst_fitness_list, fitness_distributions)

def plot_results(best_f_list, avg_f_list, worst_f_list, fitness_distributions):
    generations = list(range(len(best_f_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_f_list, label="Best Fitness")
    plt.plot(generations, avg_f_list, label="Average Fitness")
    plt.plot(generations, worst_f_list, label="Worst Fitness")
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

if __name__ == "__main__":
    main()
