import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_TARGET = "Hello World!"
GA_CROSSOVER_OPERATOR = "SINGLE"  # Default; will be updated based on user input

class GAIndividual: #Individual class for the GA 
    def __init__(self, string=None):
        self.string = string if string else self.random_string()
        self.fitness = 0

    def random_string(self): #Generates a random string of the same length as the target
        return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))

    def calculate_fitness(self): #Calculates the fitness of the individual, sum of the absolute difference between the target and the individual
        self.fitness = sum(abs(ord(self.string[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))

    def mutate(self): # Mutates by changing a random character, choosing a random character from the string and changing it by adding a random number between 0 and 90 to it
        pos = random.randint(0, len(self.string) - 1)
        delta = chr((ord(self.string[pos]) + random.randint(0, 90)) % 122)
        s = list(self.string)
        s[pos] = delta
        self.string = ''.join(s)

def init_population(): #Initializes the population, creates a list of GA_POPSIZE random individuals
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population): #Sorts the population by fitness, the fittest (individual with the lowest fitness) is at the beginning of the list
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize): #Copies the esize fittest individuals to the buffer
    buffer[:esize] = [GAIndividual(ind.string) for ind in population[:esize]]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness

# ---------- Task 4 ----------
# --- New crossover operator functions ---
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
# --- End of crossover operator functions ---

def mate(population, buffer): #Mates the population, creates a new population by mating the fittest individuals
    esize = int(GA_POPSIZE * GA_ELITRATE)
    tsize = len(GA_TARGET)

    elitism(population, buffer, esize)

    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2)
        i2 = random.randint(0, GA_POPSIZE // 2)
        
        # Instead of using a single random crossover point, use the selected operator:
        if GA_CROSSOVER_OPERATOR == "SINGLE":
            child_string = crossover_single(population[i1], population[i2])
        elif GA_CROSSOVER_OPERATOR == "TWO":
            child_string = crossover_two(population[i1], population[i2])
        elif GA_CROSSOVER_OPERATOR == "UNIFORM":
            child_string = crossover_uniform(population[i1], population[i2])
        else:
            child_string = crossover_single(population[i1], population[i2])
        
        child = GAIndividual(child_string)

        if random.random() < GA_MUTATIONRATE:
            child.mutate()

        buffer.append(child)

# ---------- Task 1 ----------
def print_generation_stats(population, generation, tick_duration, total_elapsed): #Prints the generation stats
    fitness_values = [ind.fitness for ind in population] #Gets the fitness values of the population
    best = population[0] #Gets the best individual in the population
    worst = population[-1] #Gets the worst individual in the population
    avg_fitness = sum(fitness_values) / len(fitness_values) #Calculates the average fitness of the population
    std_dev = statistics.stdev(fitness_values) #Calculates the standard deviation of the fitness of the population
    fitness_range = worst.fitness - best.fitness #Calculates the fitness range of the population

    print(f"Gen {generation}: Best = '{best.string}' (Fitness = {best.fitness})") #Prints the best individual in the population
    print(f"  Avg Fitness = {avg_fitness:.2f}") #Prints the average fitness of the population
    print(f"  Std Dev = {std_dev:.2f}") #Prints the standard deviation of the fitness of the population
    print(f"  Worst Fitness = {worst.fitness}") #Prints the worst fitness of the population
    print(f"  Fitness Range = {fitness_range}") #Prints the fitness range of the population
    print(f"  Tick Duration (sec) = {tick_duration:.4f}") #Prints the duration of the tick
    print(f"  Total Elapsed Time (sec) = {total_elapsed:.4f}") #Prints the total elapsed time
    print()

def main():
    # ---------- Task 4 ----------
    # --- User Input for Crossover Operator ---
    print("Select crossover operator:")
    print("1 - SINGLE")
    print("2 - TWO")
    print("3 - UNIFORM")
    choice = input("Enter your choice (1/2/3): ")
    global GA_CROSSOVER_OPERATOR
    if choice == "1":
        GA_CROSSOVER_OPERATOR = "SINGLE"
    elif choice == "2":
        GA_CROSSOVER_OPERATOR = "TWO"
    elif choice == "3":
        GA_CROSSOVER_OPERATOR = "UNIFORM"
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
            ind.calculate_fitness()

        sort_population(population)

        fitness_values = [ind.fitness for ind in population]
        fitness_distributions.append(fitness_values.copy())

        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)
        
        # ---------- Task 2 ----------
        tick_end = timeit.default_timer() #Ends the tick
        tick_duration = tick_end - tick_start #Calculates the duration of the tick
        total_elapsed = tick_end - start_time #Calculates the total elapsed time

        print_generation_stats(population, generation, tick_duration, total_elapsed)

        if population[0].fitness == 0:
            print(f"Converged after {generation + 1} generations.")
            break
        
        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population

    # ---------- Task 3_A ----------
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

    # ---------- Task 3_B ----------
    plt.figure(figsize=(12, 6))
    plt.boxplot(fitness_distributions, showfliers=True)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Box Plot of Fitness per Generation')
    plt.grid(True)
    plt.show()
    
    

if __name__ == "__main__":
    main()