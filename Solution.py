import random
import time
import timeit
import statistics
import matplotlib.pyplot as plt
import math  # נדרש לחישוב אנטרופיה

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

class GAIndividualARC:
    def __init__(self, program=None, grid_shape=(10, 10), target_shape=None):
        # grid_shape: גודל הגריד של החידה (למשל 10x10)
        self.grid_shape = grid_shape
        self.target_shape = target_shape  # Size of the target grid
        self.program = program if program is not None else self.random_program()
        self.fitness = None
        self.age = 0

    def random_program(self, num_ops=5):
        """Generate a random program with controlled complexity."""
        # Limit the max number of operations to prevent bloat
        num_ops = min(num_ops, 10)
        
        # For target grids that are 1x1, give preference to logical operations
        if hasattr(self, 'target_shape') and self.target_shape == (1, 1):
            op_types = ["fill", "count", "majority", "min_max", "pattern"]
            weights = [0.1, 0.3, 0.3, 0.2, 0.1]
        else:
            op_types = ["fill", "shift", "rotate", "count", "majority", "min_max", "pattern"]
            weights = [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]
        
        ops = []
        for _ in range(num_ops):
            # Weight operation types based on effectiveness
            op_type = random.choices(op_types, weights=weights, k=1)[0]
            
            if op_type == "fill":
                # Generate a more focused region rather than the entire grid
                region_size = max(2, min(self.grid_shape[0], self.grid_shape[1]) // 2)
                
                x1 = random.randint(0, self.grid_shape[0] - 1)
                y1 = random.randint(0, self.grid_shape[1] - 1)
                x2 = min(self.grid_shape[0] - 1, x1 + random.randint(0, region_size))
                y2 = min(self.grid_shape[1] - 1, y1 + random.randint(0, region_size))
                
                # Use a diverse set of colors, with preference for common ones
                color = random.choices(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    weights=[3, 2, 3, 2, 1, 1, 1, 1, 3, 1],  # 0, 2, 8 are common
                    k=1
                )[0]
                
                op = {"op": "fill", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color}
            elif op_type == "shift":
                x1 = random.randint(0, self.grid_shape[0] - 1)
                y1 = random.randint(0, self.grid_shape[1] - 1)
                x2 = min(self.grid_shape[0] - 1, x1 + random.randint(0, 3))
                y2 = min(self.grid_shape[1] - 1, y1 + random.randint(0, 3))
                
                # Smaller shifts are usually more meaningful
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
                if dx == 0 and dy == 0:
                    dx = 1  # Ensure movement
                
                op = {"op": "shift", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "dx": dx, "dy": dy}
            elif op_type == "rotate":
                # Prefer smaller rotations on specific regions
                region_size = max(2, min(self.grid_shape[0], self.grid_shape[1]) // 3)
                
                x1 = random.randint(0, self.grid_shape[0] - 2)
                y1 = random.randint(0, self.grid_shape[1] - 2)
                x2 = min(self.grid_shape[0] - 1, x1 + random.randint(1, region_size))
                y2 = min(self.grid_shape[1] - 1, y1 + random.randint(1, region_size))
                
                direction = random.choice(["cw", "ccw"])
                op = {"op": "rotate", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction}
            elif op_type == "count":
                # Count occurrences of each value and pick the most/least common
                value = random.choice([0, 2, 8])  # Common values in puzzles
                mode = random.choice(["most", "least"])
                output = random.choice([0, 2, 8])  # Output the most common as this value
                
                op = {"op": "count", "value": value, "mode": mode, "output": output}
            elif op_type == "majority":
                # Create a grid with the most/least common value
                mode = random.choice(["most", "least"])
                exclude_value = random.choice([-1, 0, 2, 8])  # -1 means no exclusion
                
                op = {"op": "majority", "mode": mode, "exclude": exclude_value}
            elif op_type == "min_max":
                # Find minimum or maximum value
                mode = random.choice(["min", "max"])
                multiplier = random.choice([1, 2, 0])  # Allows for some transformations
                
                op = {"op": "min_max", "mode": mode, "multiplier": multiplier}
            elif op_type == "pattern":
                # Look for specific patterns (e.g., 2 next to 8)
                pattern_type = random.choice(["adjacent", "corners", "border", "center"])
                value1 = random.choice([0, 2, 8])
                value2 = random.choice([0, 2, 8])
                output = random.choice([0, 2, 8])
                
                op = {"op": "pattern", "pattern": pattern_type, "value1": value1, "value2": value2, "output": output}
            
            ops.append(op)
        
        return ops

    def apply_program(self, input_grid):
        """Apply the program operations to the input grid."""
        # יישום התכנית (הרצף של הפעולות) על גריד קלט
        grid = [row[:] for row in input_grid]  # עותק של הגריד
        for op in self.program:
            if op["op"] == "fill":
                # Validate operation parameters
                x1, y1 = max(0, min(op["x1"], self.grid_shape[0]-1)), max(0, min(op["y1"], self.grid_shape[1]-1))
                x2, y2 = max(x1, min(op["x2"], self.grid_shape[0]-1)), max(y1, min(op["y2"], self.grid_shape[1]-1))
                
                # Fill operation with color
                color = op["color"]
                for i in range(x1, x2 + 1):
                    for j in range(y1, y2 + 1):
                        grid[i][j] = color
                        
            elif op["op"] == "shift":
                # Shift operation with proper handling of overlapping regions
                x1, y1 = max(0, min(op["x1"], self.grid_shape[0]-1)), max(0, min(op["y1"], self.grid_shape[1]-1))
                x2, y2 = max(x1, min(op["x2"], self.grid_shape[0]-1)), max(y1, min(op["y2"], self.grid_shape[1]-1))
                dx, dy = op["dx"], op["dy"]
                if dx == 0 and dy == 0:  # Ensure movement
                    dx = 1
                
                # Create a temporary copy of the region to shift
                temp = []
                for i in range(x1, x2 + 1):
                    temp.append(grid[i][y1:y2 + 1])
                
                # Clear the original region
                for i in range(x1, x2 + 1):
                    for j in range(y1, y2 + 1):
                        grid[i][j] = 0
                
                # Place the shifted region
                for i, row in enumerate(temp):
                    for j, val in enumerate(row):
                        ni = x1 + i + dx
                        nj = y1 + j + dy
                        if 0 <= ni < self.grid_shape[0] and 0 <= nj < self.grid_shape[1]:
                            grid[ni][nj] = val
                            
            elif op["op"] == "rotate":
                # Rotate operation with proper handling of non-square regions
                x1, y1 = max(0, min(op["x1"], self.grid_shape[0]-1)), max(0, min(op["y1"], self.grid_shape[1]-1))
                x2, y2 = max(x1, min(op["x2"], self.grid_shape[0]-1)), max(y1, min(op["y2"], self.grid_shape[1]-1))
                rows = x2 - x1 + 1
                cols = y2 - y1 + 1
                dim = min(rows, cols)
                
                # Extract the square subgrid
                subgrid = []
                for i in range(x1, x1 + dim):
                    subgrid.append(grid[i][y1:y1 + dim])
                
                # Rotate the subgrid
                if op["direction"] == "cw":
                    rotated = list(zip(*subgrid[::-1]))
                else:  # ccw
                    rotated = list(zip(*[row[::-1] for row in subgrid]))
                rotated = [list(row) for row in rotated]
                
                # Place the rotated subgrid back
                for i in range(dim):
                    for j in range(dim):
                        grid[x1 + i][y1 + j] = rotated[i][j]
                        
            elif op["op"] == "count":
                # Count occurrences of a specific value and set grid based on frequency
                target_value = op["value"]
                count = sum(row.count(target_value) for row in grid)
                total_cells = sum(len(row) for row in grid)
                
                # Determine if this value is most/least common
                other_values = set(val for row in grid for val in row if val != target_value)
                is_most_common = True
                is_least_common = True
                
                for other_val in other_values:
                    other_count = sum(row.count(other_val) for row in grid)
                    if other_count > count:
                        is_most_common = False
                    if other_count < count and other_count > 0:
                        is_least_common = False
                
                # Set output based on mode
                if (op["mode"] == "most" and is_most_common) or (op["mode"] == "least" and is_least_common):
                    # Fill entire grid with output value
                    for i in range(len(grid)):
                        for j in range(len(grid[0])):
                            grid[i][j] = op["output"]
                
            elif op["op"] == "majority":
                # Create a grid with most/least common value
                value_counts = {}
                for row in grid:
                    for val in row:
                        if val != op["exclude"]:  # Skip excluded value
                            value_counts[val] = value_counts.get(val, 0) + 1
                
                if not value_counts:  # If no values left after exclusion
                    majority_value = 0
                else:
                    # Find the most/least common value
                    if op["mode"] == "most":
                        majority_value = max(value_counts.items(), key=lambda x: x[1])[0]
                    else:  # "least"
                        majority_value = min(value_counts.items(), key=lambda x: x[1])[0]
                
                # Create a new grid with just this value
                new_grid = []
                for _ in range(1):
                    new_grid.append([majority_value])
                
                return new_grid  # Return a 1x1 grid with the majority value
                
            elif op["op"] == "min_max":
                # Find min or max value in the grid
                values = [val for row in grid for val in row]
                if not values:
                    result_value = 0
                else:
                    if op["mode"] == "min":
                        result_value = min(values)
                    else:  # "max"
                        result_value = max(values)
                    
                    # Apply multiplier
                    result_value = result_value * op["multiplier"]
                
                # Create a new grid with just this value
                new_grid = []
                for _ in range(1):
                    new_grid.append([result_value])
                
                return new_grid  # Return a 1x1 grid with the min/max value
                
            elif op["op"] == "pattern":
                pattern_found = False
                
                if op["pattern"] == "adjacent":
                    # Check if value1 is adjacent to value2
                    for i in range(len(grid)):
                        for j in range(len(grid[0])):
                            if grid[i][j] == op["value1"]:
                                # Check neighbors
                                for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                                        if grid[ni][nj] == op["value2"]:
                                            pattern_found = True
                                            break
                
                elif op["pattern"] == "corners":
                    # Check if values are in corners
                    corners = [(0,0), (0,len(grid[0])-1), (len(grid)-1,0), (len(grid)-1,len(grid[0])-1)]
                    corner_values = [grid[i][j] for i,j in corners]
                    pattern_found = op["value1"] in corner_values and op["value2"] in corner_values
                
                elif op["pattern"] == "border":
                    # Check if values are on the border
                    border_values = []
                    for i in range(len(grid)):
                        border_values.append(grid[i][0])
                        border_values.append(grid[i][len(grid[0])-1])
                    for j in range(len(grid[0])):
                        border_values.append(grid[0][j])
                        border_values.append(grid[len(grid)-1][j])
                    
                    pattern_found = op["value1"] in border_values and op["value2"] in border_values
                
                elif op["pattern"] == "center":
                    # Check if value is in center region
                    center_i = len(grid) // 2
                    center_j = len(grid[0]) // 2
                    center_region = []
                    for i in range(max(0, center_i-1), min(len(grid), center_i+2)):
                        for j in range(max(0, center_j-1), min(len(grid[0]), center_j+2)):
                            center_region.append(grid[i][j])
                    
                    pattern_found = op["value1"] in center_region or op["value2"] in center_region
                
                if pattern_found:
                    # If pattern is found, return a 1x1 grid with the output value
                    return [[op["output"]]]
        
        return grid

    def calculate_fitness(self, input_grid, target_grid):
        """Calculate fitness for ARC puzzle solutions with improved metrics."""
        try:
            output_grid = self.apply_program(input_grid)
            
            # Handle empty grids or 1x1 grids
            if not output_grid or not target_grid:
                self.fitness = float('inf')
                return self.fitness
                
            # Special case for 1x1 grid (dimension mismatch is expected)
            if len(target_grid) == 1 and len(target_grid[0]) == 1:
                # For 1x1 grid, we just need to check if any element in output matches the target
                target_value = target_grid[0][0]
                # Count occurrences of target value in output
                matches = sum(row.count(target_value) for row in output_grid)
                total_cells = sum(len(row) for row in output_grid)
                
                # If the target appears in output, consider it partially correct
                if matches > 0:
                    accuracy = matches / total_cells
                    # Apply a penalty for wrong dimensions
                    dimension_penalty = 0.2
                    self.fitness = 1 - accuracy + dimension_penalty + (len(self.program) / 20)
                else:
                    self.fitness = 2.0  # No match at all
                    
                return self.fitness
            
            # Check output dimensions match target for regular grids
            if len(output_grid) != len(target_grid) or len(output_grid[0]) != len(target_grid[0]):
                self.fitness = float('inf')  # Invalid solution
                return self.fitness
            
            # Calculate basic difference (lower is better)
            total_cells = len(target_grid) * len(target_grid[0])
            matching_cells = 0
            
            for i in range(len(target_grid)):
                for j in range(len(target_grid[0])):
                    if output_grid[i][j] == target_grid[i][j]:
                        matching_cells += 1
            
            # Core fitness: percentage of correct cells (higher is better)
            accuracy = matching_cells / total_cells
            
            # Penalize program complexity (shorter programs are better)
            complexity_penalty = len(self.program) / 20  # 5% penalty per operation
            
            # Bonus for having few operations but high accuracy
            if accuracy > 0.5 and len(self.program) < 5:
                simplicity_bonus = 0.1
            else:
                simplicity_bonus = 0
            
            # Calculate final fitness (lower is better, can be negative)
            # We transform accuracy to be minimized
            self.fitness = 1 - accuracy + complexity_penalty - simplicity_bonus
            
        except Exception as e:
            # Handle any errors in fitness calculation
            print(f"Error in fitness calculation: {e}")
            self.fitness = float('inf')
            
        return self.fitness

    def mutate(self):
        """Improved mutation strategy with anti-bloat measures."""
        if not self.program:
            # Generate a new random program if current one is empty
            self.program = self.random_program(num_ops=2)
            return
            
        # Determine mutation strategy based on program length
        # Encourage shrinking large programs and growing small ones
        if len(self.program) > 8:
            mutation_type = random.choices(
                ["change", "remove", "swap", "modify"],
                weights=[0.3, 0.4, 0.2, 0.1],  # Higher chance of removal
                k=1
            )[0]
        elif len(self.program) < 3:
            mutation_type = random.choices(
                ["change", "add", "modify"],
                weights=[0.4, 0.4, 0.2],  # Higher chance of addition
                k=1
            )[0]
        else:
            mutation_type = random.choices(
                ["change", "add", "remove", "swap", "modify"],
                weights=[0.3, 0.15, 0.15, 0.2, 0.2],
                k=1
            )[0]
        
        if mutation_type == "change":
            idx = random.randint(0, len(self.program) - 1)
            self.program[idx] = self.random_program(num_ops=1)[0]
        elif mutation_type == "add" and len(self.program) < 15:  # Cap program length
            op = self.random_program(num_ops=1)[0]
            idx = random.randint(0, len(self.program))
            self.program.insert(idx, op)
        elif mutation_type == "remove" and len(self.program) > 1:
            idx = random.randint(0, len(self.program) - 1)
            del self.program[idx]
        elif mutation_type == "swap" and len(self.program) > 1:
            idx1, idx2 = random.sample(range(len(self.program)), 2)
            self.program[idx1], self.program[idx2] = self.program[idx2], self.program[idx1]
        elif mutation_type == "modify":
            idx = random.randint(0, len(self.program) - 1)
            op = self.program[idx]
            
            # Focused modifications based on operation type
            if op["op"] == "fill":
                # For fill operations, often just change the color
                if random.random() < 0.7:
                    op["color"] = random.choices(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        weights=[3, 2, 3, 2, 1, 1, 1, 1, 3, 1],
                        k=1
                    )[0]
                else:
                    # Sometimes adjust the region slightly
                    delta = random.choice([-1, 0, 1])
                    if random.random() < 0.5:
                        op["x2"] = max(op["x1"], min(self.grid_shape[0] - 1, op["x2"] + delta))
                    else:
                        op["y2"] = max(op["y1"], min(self.grid_shape[1] - 1, op["y2"] + delta))
            elif op["op"] == "shift":
                # For shift operations, adjust the direction
                if random.random() < 0.7:
                    op["dx"] = random.choice([-1, 0, 1])
                    op["dy"] = random.choice([-1, 0, 1])
                    if op["dx"] == 0 and op["dy"] == 0:
                        op["dx"] = 1
                else:
                    # Sometimes adjust the region
                    delta = random.choice([-1, 0, 1])
                    if random.random() < 0.5:
                        op["x2"] = max(op["x1"], min(self.grid_shape[0] - 1, op["x2"] + delta))
                    else:
                        op["y2"] = max(op["y1"], min(self.grid_shape[1] - 1, op["y2"] + delta))
            elif op["op"] == "rotate":
                # For rotate operations, usually just flip direction
                if random.random() < 0.7:
                    op["direction"] = "cw" if op["direction"] == "ccw" else "ccw"
                else:
                    # Sometimes adjust the region
                    delta = random.choice([-1, 0, 1])
                    if random.random() < 0.5:
                        op["x2"] = max(op["x1"] + 1, min(self.grid_shape[0] - 1, op["x2"] + delta))
                    else:
                        op["y2"] = max(op["y1"] + 1, min(self.grid_shape[1] - 1, op["y2"] + delta))
            elif op["op"] == "count":
                # Change the target value or output value
                if random.random() < 0.5:
                    op["value"] = random.choice([0, 2, 8])
                else:
                    op["output"] = random.choice([0, 2, 8])
                    
                # Occasionally flip the mode
                if random.random() < 0.2:
                    op["mode"] = "most" if op["mode"] == "least" else "least"
            elif op["op"] == "majority":
                # Change excluded value or mode
                if random.random() < 0.7:
                    op["exclude"] = random.choice([-1, 0, 2, 8])
                else:
                    op["mode"] = "most" if op["mode"] == "least" else "least"
            elif op["op"] == "min_max":
                # Change multiplier or mode
                if random.random() < 0.7:
                    op["multiplier"] = random.choice([0, 1, 2])
                else:
                    op["mode"] = "min" if op["mode"] == "max" else "max"
            elif op["op"] == "pattern":
                # Change values or pattern type
                if random.random() < 0.4:
                    op["value1"] = random.choice([0, 2, 8])
                    op["value2"] = random.choice([0, 2, 8])
                elif random.random() < 0.7:
                    op["output"] = random.choice([0, 2, 8])
                else:
                    op["pattern"] = random.choice(["adjacent", "corners", "border", "center"])

    def crossover(self, other):
        # Enhanced crossover with multiple strategies
        if not self.program or not other.program:
            return GAIndividualARC(program=self.program, grid_shape=self.grid_shape)
            
        # Handle short programs by defaulting to simpler crossover methods
        if len(self.program) < 2 or len(other.program) < 2:
            # For very short programs, just combine them
            new_program = self.program + other.program
            return GAIndividualARC(program=new_program, grid_shape=self.grid_shape)
        
        if len(self.program) < 3 or len(other.program) < 3:
            # For short programs, use single-point crossover
            crossover_type = "single"
        else:
            crossover_type = random.choice(["single", "two", "uniform"])
        
        if crossover_type == "single":
            point1 = random.randint(0, len(self.program) - 1)
            point2 = random.randint(0, len(other.program) - 1)
            new_program = self.program[:point1] + other.program[point2:]
        elif crossover_type == "two":
            point1 = random.randint(0, len(self.program) - 2)
            point2 = random.randint(point1 + 1, len(self.program) - 1)
            point3 = random.randint(0, len(other.program) - 2)
            point4 = random.randint(point3 + 1, len(other.program) - 1)
            new_program = self.program[:point1] + other.program[point3:point4] + self.program[point2:]
        else:  # uniform
            new_program = []
            for i in range(max(len(self.program), len(other.program))):
                if i < len(self.program) and i < len(other.program):
                    new_program.append(random.choice([self.program[i], other.program[i]]))
                elif i < len(self.program):
                    new_program.append(self.program[i])
                else:
                    new_program.append(other.program[i])
                    
        return GAIndividualARC(program=new_program, grid_shape=self.grid_shape)

class GAIndividual:
    def __init__(self, string=None):
        self.string = string if string else self.random_string()
        self.fitness = 0
        self.age = 0  # ---------- Task 10: Aging attribute

    def random_string(self):
        return ''.join(chr(random.randint(32, 122)) for _ in range(len(GA_TARGET)))

    def calculate_fitness(self):
        self.fitness = sum(abs(ord(self.string[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))

    # ---------- Task 7 ----------
    def calculate_fitness_lcs(self):
        """New fitness based on LCS with offset adjustment."""
        lcs = lcs_length(self.string, GA_TARGET)
        bonus = sum(1 for i in range(len(GA_TARGET)) if self.string[i] == GA_TARGET[i])
        offset = GA_BONUS_FACTOR * len(GA_TARGET)
        self.fitness = (len(GA_TARGET) - lcs) - (GA_BONUS_FACTOR * bonus) + offset

    def mutate(self):
        pos = random.randint(0, len(self.string) - 1)
        delta = chr((ord(self.string[pos]) + random.randint(0, 90)) % 122)
        s = list(self.string)
        s[pos] = delta
        self.string = ''.join(s)

def init_population():
    return [GAIndividual() for _ in range(GA_POPSIZE)]

def sort_population(population):
    population.sort(key=lambda ind: ind.fitness)

def elitism(population, buffer, esize):
    buffer[:esize] = [GAIndividual(ind.string) for ind in population[:esize]]
    for i in range(esize):
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age  # העברת גיל

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
        child_chars.append(parent1.string[i] if random.random() < 0.5 else parent2.string[i])
    return ''.join(child_chars)

def crossover_trivial(parent1, parent2):
    return parent1.string if random.random() < 0.5 else parent2.string

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
    L = len(GA_TARGET)
    N = len(population)
    total_hamming = 0.0
    total_distinct = 0
    total_entropy = 0.0
    for j in range(L):
        freq = {}
        for ind in population:
            allele = ind.string[j]
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

# ---------- Task 1: Generation Stats, Task 8 & Task 9 Combined ----------
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
        if GA_CROSSOVER_OPERATOR == "SINGLE":
            child_string = crossover_single(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "TWO":
            child_string = crossover_two(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "UNIFORM":
            child_string = crossover_uniform(parent1, parent2)
        elif GA_CROSSOVER_OPERATOR == "TRIVIAL":
            child_string = crossover_trivial(parent1, parent2)
        else:
            child_string = crossover_single(parent1, parent2)
        child = GAIndividual(child_string)
        if random.random() < GA_MUTATIONRATE:
            child.mutate()
        buffer.append(child)

 
# --- ARC Puzzle Solving Functions ---
import json

def load_arc_puzzle(json_file):
    """Load an ARC puzzle from a JSON file. Assumes the JSON contains a key 'train' which is a list of puzzles."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def compute_diversity_metrics_arc(population):
    """Calculate diversity metrics for ARC individual population."""
    N = len(population)
    if N <= 1:
        return 0.0, 0.0, 0.0
    
    # Calculate program length diversity
    prog_lengths = [len(ind.program) for ind in population]
    avg_length = sum(prog_lengths) / N
    length_variance = sum((l - avg_length)**2 for l in prog_lengths) / N
    
    # Calculate operation type diversity
    op_types = {"fill": 0, "shift": 0, "rotate": 0}
    op_colors = {}  # For fill operations, track color distributions
    
    for ind in population:
        for op in ind.program:
            op_type = op["op"]
            op_types[op_type] = op_types.get(op_type, 0) + 1
            
            # Track color distribution for fill operations
            if op_type == "fill":
                color = op["color"]
                op_colors[color] = op_colors.get(color, 0) + 1
    
    total_ops = sum(op_types.values())
    if total_ops == 0:
        op_entropy = 0
    else:
        op_entropy = -sum((count / total_ops) * math.log2(count / total_ops) 
                         for count in op_types.values() if count > 0)
    
    # Calculate program similarity
    total_hamming = 0.0
    pairs = 0
    
    for i in range(N):
        for j in range(i+1, min(i+10, N)):  # Compare with only a few neighbors for efficiency
            # Compare program structures
            prog1, prog2 = population[i].program, population[j].program
            max_len = max(len(prog1), len(prog2))
            if max_len == 0:
                continue
                
            # Count differences in operations
            common_len = min(len(prog1), len(prog2))
            diffs = 0
            
            for k in range(common_len):
                # Two operations differ if:
                # 1. They have different types
                if prog1[k]["op"] != prog2[k]["op"]:
                    diffs += 1
                # 2. They have the same type but different key parameters
                elif prog1[k]["op"] == "fill" and prog2[k]["op"] == "fill":
                    if prog1[k]["color"] != prog2[k]["color"]:
                        diffs += 0.5  # Half penalty for different color
                elif prog1[k]["op"] == "shift" and prog2[k]["op"] == "shift":
                    if prog1[k]["dx"] != prog2[k]["dx"] or prog1[k]["dy"] != prog2[k]["dy"]:
                        diffs += 0.5  # Half penalty for different direction
                elif prog1[k]["op"] == "rotate" and prog2[k]["op"] == "rotate":
                    if prog1[k]["direction"] != prog2[k]["direction"]:
                        diffs += 0.5  # Half penalty for different rotation
            
            # Add differences due to length
            diffs += abs(len(prog1) - len(prog2))
            
            # Normalize
            norm_dist = diffs / max_len
            total_hamming += norm_dist
            pairs += 1
    
    # Average hamming distance
    avg_hamming = total_hamming / pairs if pairs > 0 else 0
    
    return avg_hamming, length_variance, op_entropy

def solve_arc_puzzle(puzzle, population_size=2048, max_iter=1000):
    global GA_MUTATIONRATE  # Declare GA_MUTATIONRATE as global
    
    input_grid = puzzle["input"]
    target_grid = puzzle["output"]
    grid_shape = (len(input_grid), len(input_grid[0]))
    target_shape = (len(target_grid), len(target_grid[0]))
    
    # Create initial population with knowledge of target shape
    population = [GAIndividualARC(grid_shape=grid_shape, target_shape=target_shape) for _ in range(population_size)]
    
    # Check if target is a single value - if so, add specialized seed programs
    if target_shape == (1, 1):
        # Add specialized "summarization" programs
        target_value = target_grid[0][0]
        
        # Add a program that just returns the target value
        simple_program = [{"op": "fill", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "color": target_value}]
        population[0] = GAIndividualARC(program=simple_program, grid_shape=grid_shape, target_shape=target_shape)
        
        # Add a program that checks if target value is most common
        count_program = [{"op": "majority", "mode": "most", "exclude": -1}]
        population[1] = GAIndividualARC(program=count_program, grid_shape=grid_shape, target_shape=target_shape)
        
        # Add a program that returns min value
        min_program = [{"op": "min_max", "mode": "min", "multiplier": 1}]
        population[2] = GAIndividualARC(program=min_program, grid_shape=grid_shape, target_shape=target_shape)
        
        # Add a program that returns max value
        max_program = [{"op": "min_max", "mode": "max", "multiplier": 0}]
        population[3] = GAIndividualARC(program=max_program, grid_shape=grid_shape, target_shape=target_shape)
        
        # Check for 0, 2, 8 values in corners
        pattern_programs = [
            {"op": "pattern", "pattern": "corners", "value1": 0, "value2": 2, "output": target_value},
            {"op": "pattern", "pattern": "corners", "value1": 0, "value2": 8, "output": target_value},
            {"op": "pattern", "pattern": "corners", "value1": 2, "value2": 8, "output": target_value}
        ]
        for i, prog in enumerate(pattern_programs):
            population[4+i] = GAIndividualARC(program=[prog], grid_shape=grid_shape, target_shape=target_shape)
    
    generation = 0
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    
    # Dynamic mutation rate
    base_mutation_rate = GA_MUTATIONRATE
    min_mutation_rate = 0.1
    max_mutation_rate = 0.5
    
    # Early stopping parameters
    patience = 50
    best_fitness = float('inf')
    no_improvement_count = 0
    
    while generation < max_iter:
        # Calculate fitness for all individuals
        for ind in population:
            ind.calculate_fitness(input_grid, target_grid)
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness)
        current_best = population[0]
        current_avg = sum(ind.fitness for ind in population) / len(population)
        
        # Update history
        best_fitness_history.append(current_best.fitness)
        avg_fitness_history.append(current_avg)
        
        # Calculate diversity metrics
        diversity = compute_diversity_metrics_arc(population)
        diversity_history.append(diversity)
        
        # Print generation stats
        print(f"ARC Gen {generation}: Best Fitness = {current_best.fitness:.4f}, Avg Fitness = {current_avg:.4f}")
        print(f"  Diversity: Hamming = {diversity[0]:.2f}, Length Variance = {diversity[1]:.2f}, Op Entropy = {diversity[2]:.2f}")
        
        # Print best program
        if generation % 10 == 0:
            print("  Best program:")
            for i, op in enumerate(current_best.program):
                print(f"    {i+1}. {op}")
                
        # Check for convergence
        if current_best.fitness == 0 or (target_shape == (1, 1) and current_best.fitness < 0.1):
            print("ARC: Converged!")
            output_grid = current_best.apply_program(input_grid)
            print("Output grid:")
            for row in output_grid:
                print(row)
            break
            
        # Early stopping check
        if current_best.fitness < best_fitness:
            best_fitness = current_best.fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"ARC: Early stopping - no improvement for {patience} generations")
                break
        
        # Dynamic mutation rate adjustment
        if no_improvement_count > patience // 2:
            GA_MUTATIONRATE = min(max_mutation_rate, base_mutation_rate * (1 + no_improvement_count / patience))
        else:
            GA_MUTATIONRATE = max(min_mutation_rate, base_mutation_rate * (1 - generation / max_iter))
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        elite_count = int(population_size * GA_ELITRATE)
        new_population.extend(population[:elite_count])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Select parents using tournament selection
            tournament_size = max(2, int(population_size * 0.1))
            candidates = random.sample(population, tournament_size)
            parent1 = min(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, tournament_size)
            parent2 = min(candidates, key=lambda x: x.fitness)
            
            # Create child through crossover
            child = parent1.crossover(parent2)
            
            # Apply mutation
            if random.random() < GA_MUTATIONRATE:
                child.mutate()
                
            new_population.append(child)
        
        # Update population
        population = new_population
        generation += 1
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([d[0] for d in diversity_history], label='Program Distance')
    plt.plot([d[1] for d in diversity_history], label='Length Variance')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Population Diversity')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([d[2] for d in diversity_history], label='Operation Entropy')
    plt.xlabel('Generation')
    plt.ylabel('Entropy (bits)')
    plt.title('Operation Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return population[0], best_fitness_history

def visualize_grid_transformation(input_grid, output_grid, target_grid, title="Grid Transformation"):
    """Visualize the input grid, output grid, and target grid side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot input grid
    axes[0].imshow(input_grid, cmap='viridis')
    axes[0].set_title('Input Grid')
    axes[0].axis('off')
    
    # Plot output grid
    axes[1].imshow(output_grid, cmap='viridis')
    axes[1].set_title('Output Grid')
    axes[1].axis('off')
    
    # Plot target grid
    axes[2].imshow(target_grid, cmap='viridis')
    axes[2].set_title('Target Grid')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    print("Select mode:")
    print("1 - Run GA for Hello World")
    print("2 - Run GA for ARC Puzzle Solving")
    mode_choice = input("Enter your choice (1/2): ")
    if mode_choice == "2":
        json_file = input("Enter the ARC puzzle JSON file path: ")
        data = load_arc_puzzle(json_file)
        
        # Determine which puzzles to solve
        print(f"Loaded {len(data['train'])} training puzzles.")
        puzzle_choice = input("Enter puzzle number to solve (or 'all' for all puzzles): ")
        
        puzzles_to_solve = []
        if puzzle_choice.lower() == 'all':
            puzzles_to_solve = list(range(len(data['train'])))
        else:
            try:
                puzzle_idx = int(puzzle_choice) - 1  # Convert to 0-indexed
                if 0 <= puzzle_idx < len(data['train']):
                    puzzles_to_solve = [puzzle_idx]
                else:
                    print(f"Invalid puzzle number. Using first puzzle.")
                    puzzles_to_solve = [0]
            except ValueError:
                print(f"Invalid input. Using first puzzle.")
                puzzles_to_solve = [0]
        
        # Solve each selected puzzle
        for puzzle_idx in puzzles_to_solve:
            print(f"\n{'='*50}")
            print(f"Solving puzzle {puzzle_idx + 1}/{len(data['train'])}")
            print(f"{'='*50}\n")
            
            sample_puzzle = data["train"][puzzle_idx]
            
            # Display input and target grids
            print("Input Grid:")
            for row in sample_puzzle["input"]:
                print(row)
            print("\nTarget Grid:")
            for row in sample_puzzle["output"]:
                print(row)
            print("\nSolving...\n")
            
            # Run the genetic algorithm
            best_solution, fitness_history = solve_arc_puzzle(sample_puzzle)
            output_grid = best_solution.apply_program(sample_puzzle["input"])
            
            # Print the solution program
            print("\nBest Solution Program:")
            for i, op in enumerate(best_solution.program):
                print(f"{i+1}. {op['op']} - {op}")
            
            # Check if solution is correct
            is_correct = all(output_grid[i][j] == sample_puzzle["output"][i][j] 
                          for i in range(len(sample_puzzle["output"])) 
                          for j in range(len(sample_puzzle["output"][0])))
            
            print(f"\nSolution is {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"Final fitness: {best_solution.fitness:.4f}")
            
            # Visualize the transformation
            visualize_grid_transformation(
                sample_puzzle["input"],
                output_grid,
                sample_puzzle["output"],
                f"ARC Puzzle {puzzle_idx + 1} Solution"
            )
        
        return
    
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
            print(f"Converged after {generation + 1} generations.")
            break
        buffer.clear()
        mate(population, buffer)
        population, buffer = buffer, population
        # ---------- Task 10: Apply Aging ----------
        population = apply_aging(population)
        generation += 1

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
