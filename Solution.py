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

def mate(population, buffer): #Mates the population, creates a new population by mating the fittest individuals
    esize = int(GA_POPSIZE * GA_ELITRATE)
    tsize = len(GA_TARGET)

    elitism(population, buffer, esize)

    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2)
        i2 = random.randint(0, GA_POPSIZE // 2)
        spos = random.randint(0, tsize - 1)

        child_string = population[i1].string[:spos] + population[i2].string[spos:]
        child = GAIndividual(child_string)

        if random.random() < GA_MUTATIONRATE:
            child.mutate()

        buffer.append(child)

# ---------- Task 1 ----------
def print_generation_stats(population, generation): #Prints the generation stats
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
    print()

def main():
    random.seed(time.time())
    start_time = timeit.default_timer()

    population = init_population()
    buffer = []
    best_fitness_list = []
    avg_fitness_list = []
    worst_fitness_list = []

    for generation in range(GA_MAXITER):
        tick_start = timeit.default_timer()
        for ind in population:
            ind.calculate_fitness()

        sort_population(population)

        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_list.append(best_fitness)
        avg_fitness_list.append(avg_fitness)
        worst_fitness_list.append(worst_fitness)

        print_generation_stats(population, generation)

        if population[0].fitness == 0:
            print(f"Converged after {generation + 1} generations.")
            break
        
        # ---------- Task 2 ----------

        tick_end = timeit.default_timer() #Ends the tick
        tick_duration = tick_end - tick_start #Calculates the duration of the tick
        total_elapsed = tick_end - start_time #Calculates the total elapsed time
        print(f"Tick Duration (sec) = {tick_duration:.4f}") #Prints the duration of the tick
        print(f"Total Elapsed Time (sec) = {total_elapsed:.4f}") #Prints the total elapsed time

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

if __name__ == "__main__":
    main()