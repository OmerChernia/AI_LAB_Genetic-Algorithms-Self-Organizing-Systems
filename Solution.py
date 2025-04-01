import random
import time

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

def print_best(population, generation): #Prints the best individual in the population
    best = population[0]
    print(f"Gen {generation}: Best = '{best.string}' (Fitness = {best.fitness})")

def main():
    random.seed(time.time())

    population = init_population()
    buffer = []

    for generation in range(GA_MAXITER):
        for ind in population:
            ind.calculate_fitness()

        sort_population(population)
        print_best(population, generation)

        if population[0].fitness == 0:
            break

        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population

if __name__ == "__main__":
    main()