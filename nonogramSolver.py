"""
Lior Sinai, 9 October 2020

Solves a nonogram puzzles

See https://en.wikipedia.org/wiki/Nonogram

Based off the codililty coding question in nonogram.py, but optimsied and generalised

Notation
- # black
- . space/white
- ? unknown 

2 solutions methods
- constraint propagation
- exhaustive search (very slow without the former)

"""

from itertools import permutations
import re

import time
import matplotlib.pyplot as plt
import matplotlib.colors

from integerPartitions import integer_partitions, unique_perm_partitions
from matcher import Match, find_match

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

class Nonogram():
    def __init__(self, runs_row, runs_col):
        self.n_rows = len(runs_row)
        self.n_cols = len(runs_col)
        self.runs_row = runs_row
        self.runs_col = runs_col
        self.grid = [[EITHER] * self.n_cols for i in range(self.n_rows)]
        
        self.solutions = []        
        self.calls = 0

    @property
    def num_solutions(self):
        return len(self.solutions)

    @property
    def progress(self):
        "the fraction of cells completed (correct or not)"
        return 1 - sum(row.count(EITHER) for row in self.grid)/(self.n_rows * self.n_cols)

    def set_grid(self, grid):
        self.grid = [row[:] for row in grid]


    def make_box(self, symbols="x#.?"):
        "For display purposes. Includes instructions and grid"
        row_max_length = len(max(self.runs_row, key=len))
        col_max_length = len(max(self.runs_col, key=len))

        box = []

        # make column instructions
        for i in range(col_max_length):
            row = ["-"] * (row_max_length + self.n_cols)
            for j in range(self.n_cols):
                run = self.runs_col[j]
                if len(run) >= col_max_length - i:
                    row[row_max_length + j] = str(run[::-1][col_max_length - i - 1])
            box.append(row)
        # make row instructions
        for i, row in enumerate(self.grid):
            run = self.runs_row[i]
            row = ["-"] * (row_max_length - len(run)) + list(map(str, run)) + [symbols[x] for x in row]
            box.append(row)
        
        return box

    def show_grid(self, show_instructions=True, symbols="x#.?"):
        if show_instructions:
            grid = self.make_box(symbols) 
        else:
            grid = []
            for row in self.grid:
                grid.append([symbols[x] for x in row]) # x-> something is wrong, #->black, .->white, ?->was never assigned
        for row in grid:
            for symbol in row:
                print("{:2}".format(symbol), end='')
            print("")


    def is_complete(self, grid=None):
        if grid is None:
            grid = self.grid
        for arr, run in zip(grid + list(zip(*grid)), self.runs_row + self.runs_col):
            if not self.is_valid_line(arr, run):
                return False    
        return True

    def _get_sequence(self, arr):
        white = True
        sequence = []
        positions = [] #where the sequences start and end
        for idx, color in enumerate(arr):
            if color == BLACK and white:
                sequence.append(1)
                positions.append([idx, idx])
                white = False
            elif color == BLACK and not white:
                sequence[-1] += 1
                positions[-1][1] = idx
            else: #  color == WHITE or EITHER
                white = True 

        return sequence, positions

    def is_valid_line(self, arr, target):
        sequence, _ = self._get_sequence(arr)
        return sequence == list(target)

    def solve(self):
        "Do constraint propagation before exhaustive search. Based on the RosettaCode code"
        def initialise(length, runs):
            """If any sequence x in the run is greater than the number of free whites, some these values will be fixed"""
            # The first run of fix_row() or fix_col() will find this anyway. But this is faster
            arr = [EITHER] * length
            free_whites = length - sum(runs) - (len(runs) - 1)  # remaining whites to place
            j = 0 # current position
            for x in runs:
                if x > free_whites: # backfill s 
                    for c in range(j + free_whites, j + x): 
                        arr[c] = BLACK 
                if (free_whites == 0) and (j + x < length):
                    arr[j + x] = WHITE # can place a white too
                j += x + 1
            return arr

        def fits(a, b):
            """
            Use binary to represent white and black
            01 -> black, 10 -> white, 11 -> either
            black & white == 0 -> invalid
            black & black == black
            white & white == white
            black & 3 == either 
            white & 3 == either
            """
            return all(x & y for x, y in zip(a, b))


        def generate_sequences(fixed, runs):
            """
            Generates valid sequences. 
            """
            length = len(fixed)
            n = len(runs)
            sequence_whites = [0] + [1] * (n - 1) + [0] # at least one white in between each
            free_whites = length - sum(runs) - (n - 1)  # remaining whites to place
            
            possible = [] # all possible permutations of the run

            # find all ways to place remaining free whites
            for partition in unique_perm_partitions(free_whites, len(sequence_whites)): # unique partitions
                arr = []
                for n_white, n_free, n_black in zip(sequence_whites, partition, runs):
                    arr += [WHITE] * (n_white + n_free) + ([BLACK]  * n_black) 
                # place last set of whites
                arr += [WHITE] * (sequence_whites[n] + partition[n])
                if fits(arr, fixed): # there are no conflicts
                    possible.append(arr)

            return possible

        def find_allowed(possible_arrays):
            """Finds all allowed value in the set of possible arrays
            This is done with binary OR. Values that are always 1 or 2 will never change. If either, they will become a 3
            If they remain 0 -> there is no solution
            """
            allowed = [0] * len(possible_arrays[0])
            for array in possible_arrays:
                allowed = [x | y for x, y in zip(allowed, array)] 
            return allowed

        def fix_row(i):
            fixed = self.grid[i]
            possible_rows[i] = [row for row in possible_rows[i] if fits(fixed, row)] # reduce the amount of possible rows
            allowed = find_allowed(possible_rows[i])
            for j in range(self.n_cols):
                if fixed[j] != allowed[j]:
                    columns_to_edit.add(j)
                    self.grid[i]= allowed

        def fix_col(j):
            fixed = [self.grid[i][j] for i in range(self.n_rows)]
            possible_cols[j] = [col for col in possible_cols[j] if fits(fixed, col)] # reduce the amount of possible cols
            allowed = find_allowed(possible_cols[j])
            for i in range(self.n_rows):
                if fixed[i] != allowed[i]:
                    rows_to_edit.add(i)
                    self.grid[i][j] = allowed[i]

        #ax = plot_nonogram(game.grid, save=save, filename="solving_sweep_0")
        for i in range(self.n_rows):
           self.grid[i] = initialise(self.n_cols, self.runs_row[i])
        for j in range(self.n_cols):
            col = initialise(self.n_rows, self.runs_col[j])
            for i in range(self.n_rows):
                self.grid[i][j] = col[i]
        #plot_nonogram(game.grid, ax=ax, save=save, filename="solving_sweep_1")

        possible_rows = [generate_sequences(self.grid[i], self.runs_row[i]) for i in range(self.n_rows)]
        possible_cols = []
        for j in range(self.n_cols):
            col = [self.grid[i][j] for i in range(self.n_rows)]
            possible_cols.append(generate_sequences(col, self.runs_col[j]))

        print("initial")
        print("possible rows: {}\npossible columns: {}".format([len(x) for x in possible_rows], [len(x) for x in possible_cols]))
        print("summary: {} possible rows and {} possible columns".format(sum([len(x) for x in possible_rows]), 
                                                                        sum([len(x) for x in possible_cols])))

        # rows, columns for constraint propagation to be applied
        rows_to_edit = set(range(self.n_rows))
        columns_to_edit = set(range(self.n_cols))

        sweeps = 2 # include initialising
        while columns_to_edit:
            # constraint propagation
            for j in columns_to_edit:
                fix_col(j)
            sweeps += 1
            #plot_nonogram(game.grid, ax=ax, save=save, filename="solving_sweep_{}".format(sweeps))
            columns_to_edit = set()
            for i in rows_to_edit:
                fix_row(i)
            sweeps += 1
            #plot_nonogram(game.grid, ax=ax, save=save, filename="solving_sweep_{}".format(sweeps))
            
            rows_to_edit = set()
            
        print("\nconstraint propagation done in {} rounds".format(sweeps))

        solution_found = all(self.grid[i][j] in (1, 2) for j in range(self.n_cols) for i in range(self.n_rows)) # might be incorrect
        if solution_found:
            print("Solution is unique") # but could be incorrect!
        else:
            print("Solution may not be unique, doing exhaustive search:")
        
        def try_all(i = 0):
            if i >= self.n_rows:
                for j in range(self.n_cols):
                    col = [row[j] for row in grid]
                    if col not in possible_cols[j]:
                        return 0
                self.set_grid(grid)
                self.show_grid()
                print("")
                return 1
            sol = 0
            for row in possible_rows[i]:
                grid[i] = row
                sol += try_all(i + 1)
            return sol
        
        if not solution_found:
            grid_original = [row[:] for row in self.grid]
            grid = [row[:] for row in self.grid]
            num_solutions = try_all(i=0)
            if num_solutions == 0:
                print("no solutions found")
            elif num_solutions == 1:
                print("unique solution found")
            else: # num_solutions > 1:
                 print("{} solutions found".format(num_solutions))
            self.set_grid(grid_original)
        
        print("\nafter solving:")
        print("possible rows: {}\npossible columns: {}".format([len(x) for x in possible_rows], [len(x) for x in possible_cols]))
        print("summary: {} possible rows and {} possible columns".format(sum([len(x) for x in possible_rows]), 
                                                                        sum([len(x) for x in possible_cols])))
        

    def solve_fast(self):
        def simple_fuller(arr, runs):
            """ fill in gaps and whites ending sequences. The overlap algorithm might miss these"""
            i, k = 0, 0
            on_black = 0
            allowed = arr[:]
            min_length = runs[0]
            for i in range(len(arr)):
                if arr[i] == WHITE:
                    if on_black > 0:
                        k += 1 # move to the next pattern
                    on_black = 0
                elif arr[i] == BLACK:
                    on_black += 1
                else: # arr[i] == EITHER
                    if (0 < on_black < runs[k]): # this must be part of this sequence
                        allowed[i] = BLACK
                        on_black += 1
                    elif on_black == 0 and k > 0 and i >0 and arr[i-1] == BLACK: # this must be a white ending a sequence
                        allowed[i] = WHITE    
                    elif k >= len(runs): # only whites left
                        allowed[i] = WHITE         
                    else:
                        break # too many unknowns
            return allowed

        def changer_sequence(vec):
            """ convert to ascending sequence """
            counter = int(vec[0] == BLACK)
            prev = vec[0]
            sequence = [counter]
            for x in vec[1:]:
                counter += (prev != x) # increase by one every time a new sequence starts
                sequence.append(counter)
                prev = x
            return sequence

        def overlap(a, b):
            out = []
            for x, y in zip(changer_sequence(a), changer_sequence(b)):
                if x==y:
                    if (x+2) % 2 == 0:
                        out.append(WHITE) 
                    else:
                        out.append(BLACK)
                else:
                    out.append(EITHER)
            return out

        def left_rightmost_overlap(arr, runs):
            """Returns the overlap between the left-most and right-most fitting sequences"""
            left = find_match(arr, runs)
            right = find_match(arr[::-1], runs[::-1])
            if left.is_match and right.is_match:
                allowed = overlap(left.match, right.match[::-1])
            else:
                allowed = arr[:]
            return allowed
                
        def fix_row(i):
            row = self.grid[i]
            allowed1 = left_rightmost_overlap(row, self.runs_row[i])
            allowed2 = simple_fuller(row, self.runs_row[i])
            allowed3 = simple_fuller(row[::-1], self.runs_row[i][::-1])[::-1] # going from right
            allowed = [x & y & z for x, y, z in zip(allowed1, allowed2, allowed3)]
            for j in range(self.n_cols):
                if row[j] != allowed[j] and allowed[j]!=EITHER:
                    columns_to_edit.add(j)
                    self.grid[i]= allowed
        
        def fix_col(j):
            col = [self.grid[i][j] for i in range(self.n_rows)]
            allowed1 = left_rightmost_overlap(col, self.runs_col[j])
            allowed2 = simple_fuller(col, self.runs_col[j])
            allowed3 = simple_fuller(col[::-1], self.runs_col[j][::-1])[::-1] # going from right
            allowed = [x & y & z for x, y, z in zip(allowed1, allowed2, allowed3)]
            for i in range(self.n_rows):
                if col[i] != allowed[i] and allowed[i]!=EITHER:
                    rows_to_edit.add(i)
                    self.grid[i][j] = allowed[i]
            
        # rows, columns for constraint propagation to be applied
        rows_to_edit = set()
        columns_to_edit = set(range(self.n_cols))

        for i in range(self.n_rows):
            fix_row(i)

        rounds = 1 # includie initialise
        while columns_to_edit:
            rounds += 1
            #constraint propagation
            for j in columns_to_edit:
                fix_col(j)
            columns_to_edit = set()
            for i in rows_to_edit:
                fix_row(i)
            rows_to_edit = set()
        print("\nconstraint propagation done in {} rounds".format(rounds))



def encode_puzzle(filename, runs_row, runs_col, description="") -> None:
    "Encode a puzzle in a text file in strings of letters"
    with open(filename, "a+") as file:
        file.write("\n--" + description)
        for runs in (runs_row, runs_col):
            s = "\n" + " ".join([''.join(map(lambda num: chr(num + ord('A') - 1), seq)) for seq in runs])
            file.write(s)


def plot_nonogram(grid, ax=None, save=False, filename="nonogoram"):
    if not ax:
        _, ax = plt.subplots()
    n_rows, n_cols = len(grid), len(grid[0])

    # custom color map
    cmap = matplotlib.colors.ListedColormap(['black', 'white', 'blue'])
    boundaries = [0, BLACK+0.01, WHITE+0.01, EITHER]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # plot
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

    # set grid lines. Majors must not overlap with minors
    # intervals = 10
    # step_major = max(min(n_rows, n_rows) // intervals, 1) # space out numbers for larger grid
    # if step_major % 2 == 1:
    #     step_major += 1 # always have an even step
    step_major = 5
    if min(n_rows, n_rows) // step_major < step_major:
            step_major = 1
    else:
        while min(n_rows, n_rows) // step_major > step_major:
            step_major += step_major
    

    ax.set_xticks([x-0.5 for x in list(range(1, n_cols + 1))], minor=True)
    ax.set_xticks(list(range(0, n_cols, step_major)), minor=False)
    ax.set_yticks([x-0.5 for x in list(range(1, n_rows + 1))], minor=True)
    ax.set_yticks(list(range(0, n_rows, step_major)), minor=False) 
    plt.grid(which="minor", linewidth=1.1, color="k", alpha=0.7)

    if save:
        fig = plt.gcf()
        fig.savefig(filename)

    return ax

if __name__ == '__main__':
    # the wikipedia W
    runs_row = [(8, 7, 5, 7),(5, 4, 3, 3),(3, 3, 2, 3),(4, 3, 2, 2),(3, 3, 2, 2),(3, 4, 2, 2),(4, 5, 2),(3, 5, 1),(4, 3, 2),(3, 4, 2),(4, 4, 2),(3, 6, 2),(3, 2, 3, 1),(4, 3, 4, 2),(3, 2, 3, 2),(6, 5),(4, 5),(3, 3),(3,3), (1, 1)]
    runs_col = [(1,),(1,),(2,),(4,),(7,),(9,),(2, 8),(1, 8),(8,),(1, 9),(2, 7),(3, 4),(6, 4),(8, 5),(1, 11),(1, 7),(8,),(1, 4, 8),(6, 8),(4, 7),(2, 4),(1, 4),(5,),(1, 4),(1,5),(7,),(5,),(3,),(1,),(1,)]

    # elephant
    runs_row = [(3,),(4,2),(6,6),(6,2,1),(1,4,2,1), (6,3,2),(6,7),(6,8),(1,10),(1,10), (1,10),(1,1,4,4),(3,4,4),(4,4),(4,4)]
    runs_col = [(1,),(11,),(3,3,1),(7,2),(7,), (15,), (1,5,7),(2,8),(14,),(9,), (1,6),(1,9),(1,9),(1,10),(12,)]

    # # ## chess board, multiple solutions
    #runs_row = [(1,), (1,), (1,)]
    #runs_col = [(1,), (1,), (1,)]

    # # Bonus
    #runs_row = [(3,),(1,),(1,),(1,1),(1,),(1,1,4,1),(2,1,1,1,4),(1,1,1,1,1,1),(1,1,1,1,1),(2,1,1,1,1),(1,4,2,1,1,1),(1,3,1,4,1)]
    #runs_col = [(0,),(1,1,1),(1,5),(7,1),(1,),(2,),(1,),(1,),(1,),(0,),(2,),(1,6),(0,),(6,),(1,1),(1,1),(1,1),(6,),(0,),(1,),(7,),(1,),(1,),(1,),(0,)]

    # # aeroplane -> solve fast doesn't work. https://www.youtube.com/watch?v=MZQDDzzRBvI
    #runs_col = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,2,1],[3,3],[3,4],[2,1,2]]
    #runs_row = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,4],[3,1,2],[3,2,2],[2,1,1]]

    ## https://www.researchgate.net/publication/290264363_On_the_Difficulty_of_Nonograms
    ## Batenburg construction -> requires 120 sweeps
    five_strip1 = [(1,2,)+ 6*(1,), (2,) + 6 * (1,) + (2,), (1,), (2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    three_strip = [(1,), (18,), (1,1)]
    five_strip2 = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1),(1,),(2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    two_strip   = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1)]
    #runs_row = five_strip1 + three_strip + five_strip2 + three_strip + two_strip
    #runs_col = [(3,2,3,2,1),(1,1,2,1,1,2,1)] + [(1,)*7 for _ in range(15)] + [(2,1,3,1,3)]
    ## small but constraint propagation alone won't solve 
    #runs_row = [(4,),(4,),(1,),(1,1),(1,)] # not possible to solve with this solver
    #runs_col = [(1,),(2,1),(2,1),(2,1),(1,1)] # not possible to solve with this solver

    game = Nonogram(runs_row, runs_col)

    ## set solution
    #game.set_grid(solution)

    ## solve game
    start_time = time.time()
    #game.solve_brute()
    #game.solve()
    game.solve_fast()
    end_time = time.time()

    game.show_grid(show_instructions=True, symbols="x#.?")
    print(game.is_complete(), "{:.2f}%%".format(game.progress*100))
    print("time taken: {:.5f}s".format(end_time - start_time))

    plot_nonogram(game.grid)

    if 1==1:
        start_time = time.time()
        #filename = 'rosetta_code_puzzles.txt'
        filename = "activity_workshop_puzzles.txt"  ##  https://activityworkshop.net/puzzlesgames/nonograms 
        with open(filename) as file:
            lines = file.read().split("\n")
            for i in range(0, len(lines), 3):
                s = lines[i + 1].strip().split(" ")
                runs_row = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]
                s = lines[i + 2].strip().split(" ")
                runs_col = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]

                game = Nonogram(runs_row, runs_col)
                game.solve_fast()
                game.show_grid(show_instructions=True, symbols="x#.?")
                print(game.is_complete(), "{:.2f}%%".format(game.progress*100))

                plot_nonogram(game.grid)

        print("time taken: {:.4f}s".format(time.time() - start_time))

    plt.show()