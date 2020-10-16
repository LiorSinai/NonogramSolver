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

    def set_grid(self, grid):
        self.grid = [row[:] for row in grid]


    def make_box(self, symbols="x#.?"):
        "For display purposes. Includes instructions and grid"
        row_max_length = len(max(runs_row, key=len))
        col_max_length = len(max(runs_col, key=len))

        box = []

        # make column instructions
        for i in range(col_max_length):
            row = ["-"] * (row_max_length + self.n_cols)
            for j in range(self.n_cols):
                run = runs_col[j]
                if len(run) >= col_max_length - i:
                    row[row_max_length + j] = str(run[::-1][col_max_length - i - 1])
            box.append(row)
        # make row instructions
        for i, row in enumerate(self.grid):
            run = runs_row[i]
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

    def solve_brute(self):
        "Solve the nonogram using only exhaustive search -> very slow"
        def find_solution(grid, k=0):
            self.calls += 1
            if (self.calls) % 100 == 0:
                print(self.calls, k)

            if k == self.n_rows:
                if self.is_complete(grid):
                    print("Solution found at {} calls".format(self.calls))
                    self.set_grid(grid)
                    self.solutions.append(grid)
                return
            
            # place a valid solution along a row
            sequence_black = self.runs_row[k]
            n = len(sequence_black)
            sequence_whites = [0] + [1] * (n - 1) + [0] # at least one white in between each
            free_whites = self.n_cols - sum(sequence_black) - (n- 1)  # remaining whites to place

            grid_next = [row[:] for row in grid]

            # find all ways to place remaining free whites
            for partition in unique_perm_partitions(free_whites, len(sequence_whites)): # unique partitions
                # add perms to sequence whites and chain with black
                row = []
                for i in range(n):
                    # place whites
                    row += [WHITE] * (sequence_whites[i] + partition[i])
                    # place blacks
                    row += [BLACK]  * sequence_black[i]
                # place last set of whites
                row += [WHITE] * (sequence_whites[n] + partition[n])
                grid_next[k] = row[:]
                # check does not violate any column rules
                if self.is_valid_partial_columns(grid_next, k):
                    # go one level deeper
                    find_solution(grid_next, k=k + 1)

        grid = [row[:] for row in self.grid] 
        find_solution(grid, k=0)

    def is_valid_partial_columns(self, grid, end_row):
        columns = list(zip(*grid[:end_row + 1]))
        for j, col in enumerate(columns):
            if not self.is_valid_partial(col, self.runs_col[j]):
                return False
        return True
    
    def is_valid_partial(self, arr, target):   
        "do placed black squares follow the rules so far?"
        sequence, _ = self._get_sequence(arr)
        
        m = len(sequence)
        if m == 0:
            return True # no black squares placed so far
        elif m > len(target):
            return False # too many gaps
        elif m == 1:
            return sequence[0] <= target[0] 
        else:
            return sequence[:m-1] == list(target[:m-1]) and sequence[m-1] <= target[m-1] 

    def solve(self):
        "Do constraint propogation before exhaustive search. Based on the RosettaCode code"
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

        # rows, columns for constraint propogation to be applied
        rows_to_edit = set(range(self.n_rows))
        columns_to_edit = set(range(self.n_cols))

        sweeps = 2 # includie initialising
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
            
        print("\nconstraint propogation done in {} rounds".format(sweeps))

        solution_found = all(self.grid[i][j] in (1, 2) for j in range(self.n_cols) for i in range(self.n_rows))
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

        def changer_sequence(vec):
            counter = int(vec[0] == BLACK)
            prev = vec[0]
            sequence = [counter]
            for x in vec[1:]:
                counter += (prev != x) # increase by one every time a new sequence starts
                sequence.append(counter)
                prev = x
            return sequence

        def overlap(a, b):
            # convert to ascending sequence  
            seq_a = changer_sequence(a)
            seq_b = changer_sequence(b)
            out = []
            for x, y in zip(seq_a, seq_b):
                if x==y:
                    if (x+2) % 2 == 0:
                        out.append(WHITE) 
                    else:
                        out.append(BLACK)
                else:
                    out.append(EITHER)
            return out
                
        def fix_row(i):
            # find the left-most sequence that fits
            # minimum is with one zero
            row = self.grid[i]
            left = find_match(row, runs_row[i], start=True)
            right = find_match(row[::-1], runs_row[i][::-1], start=True)
            if left.is_match and right.is_match:
                allowed = overlap(left.match, right.match[::-1])
                for j in range(self.n_cols):
                    if row[j] != allowed[j] and allowed[j]!=EITHER:
                        columns_to_edit.add(j)
                        self.grid[i]= allowed
        
        def fix_col(j):
            # find the left-most sequence that fits
            # minimum is with one zero
            col = [self.grid[i][j] for i in range(self.n_rows)]
            left = find_match(col, runs_col[j], start=True)
            right = find_match(col[::-1], runs_col[j][::-1], start=True)
            if left.is_match and right.is_match:
                allowed = overlap(left.match, right.match[::-1])
                for i in range(self.n_rows):
                    if col[i] != allowed[i] and allowed[i]!=EITHER:
                        rows_to_edit.add(i)
                        self.grid[i][j] = allowed[i]
            
        # rows, columns for constraint propogation to be applied
        rows_to_edit = set()
        columns_to_edit = set(range(self.n_cols))

        # for i in range(self.n_rows):
        #     fix_row(i)
        for i in range(self.n_rows):
           self.grid[i] = initialise(self.n_cols, self.runs_row[i])

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
        print("\nconstraint propogation done in {} rounds".format(rounds))


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
    ax.set_xticks([x-0.5 for x in list(range(1, n_cols + 1))], minor=True)
    ax.set_xticks(list(range(0, n_cols)), minor=False)
    ax.set_yticks([x-0.5 for x in list(range(1, n_rows + 1))], minor=True)
    ax.set_yticks(list(range(0, n_rows)), minor=False) 
    plt.grid(which="minor", linewidth=1.1, color="k", alpha=0.7)

    if save:
        fig = plt.gcf()
        fig.savefig(filename)

    return ax

if __name__ == '__main__':
    # the wikipedia W -> very slow on solve_brute()
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

    # # aeroplane -> solve fast doesn't work??
    #runs_col = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,2,1],[3,3],[3,4],[2,1,2]]
    #runs_row = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,4],[3,1,2],[3,2,2],[2,1,1]]

    ## very large 100 x 70 -> my computer can't deal
    #runs_row = [[12,35,19],[1,18,17],[1,20,20,13],[4,22,22],[1,5,23,23,9],[2,24,24],[4,25,25],[2,4,25,25],[3,26,26],[2,2,26,26],[4,2,26,26],[5,1,2,26,26],[1,1,2,1,26,26],[1,1,1,5,11,14,11,5],[1,1,2,2,6,5,2,5,6,3,2],[1,1,1,6,2,7,9],[1,2,26],[2,24,1],[4,6,2,11,3,6],[8,2,2,2,9,2,5],[10,3,3,2,2,6,4],[4,7,13,5,6,2,5,2,2,5],[2,2,4,2,1,6,1,2,6,4,6,5],[7,8,4,1,1,3,1,4,1,2,6,2,2,2,5],[6,2,2,4,5,1,4,1,2,7,1,1,1,5],[2,2,6,2,4,1,1,3,2,1,6,1,2,7,1,1,1,6],[5,2,2,3,2,3,3,4,5,5,2,4,2,2,2,3],[1,1,1,4,2,4,6,3,3,2,4,2,2,1,4],[3,2,1,1,4,1,14,2,2,2,3,2,2,1,1,4],[2,3,4,1,4,5,2,6,2,3,1,1,4],[1,5,9,5,5,2,2,3,4,4],[1,6,1,4,2,4,1,3,1,4,2,3,1,1,3],[1,4,2,3,2,1,1,1,3,2,1,2,4,13,4],[2,4,1,1,1,1,1,4,2,7,3,3,2,2],[3,6,1,1,4,3,2,3,2,2,2,1,12,2],[2,2,2,2,1,11,2,2,2,2,2,10,1,3,3],[2,2,2,2,2,2,10,2,2,5,1,2,8,1,1,1,1,1],[2,9,1,1,2,1,1,5,2,6,1,6,2,1,1,1,1],[1,10,2,2,2,1,2,2,2,2,1,1,2,2,1],[1,10,1,2,3,2,1,1,2,2,1,1,1,1],[2,9,1,2,1,1,2,1,7,3,3,2,1,1,2],[4,12,1,1,2,1,1,1,3,5,1,3,4],[2,8,1,10,1,1,1,8,2,1,1,1,3],[2,8,2,7,2,2,6,3,6,1,1,2,1],[3,8,8,1,1,7,9,4,2],[3,10,1,2,2,2,8,7,3,1,1,1],[4,18,3,14,13,7,2,2,2,2],[7,12,1,1,2,2,2,3,7,6,2,7,2],[20,6,4,13,2,8,8,2,1,2,1],[20,1,1,2,1,1,2,23,2,2,1],[4,2,3,7,5,11,2,21,2,2,2],[1,1,1,1,1,1,2,1,2,21,3,1,2],[1,1,1,1,1,1,1,1,2,19,3,1,2],[6,1,14,6,11,2,19,6,2],[6,1,14,6,10,3,3,19,2,3],[6,1,15,6,9,1,1,1,1,19,1,2],[6,2,15,6,9,1,1,1,1,19,1,1,3],[6,1,15,6,9,1,2,2,1,19,1,1,2],[6,1,15,6,8,1,3,1,19,2,1,2],[5,1,15,7,8,3,2,19,1,1,2],[1,1,1,1,1,1,2,1,10,19,1,1,2],[1,1,1,2,1,1,2,1,2,1,19,2,1,1],[1,1,1,1,2,1,2,1,1,2,19,2,1,1],[1,1,1,1,2,3,2,1,4,21,2,1,1],[1,1,2,1,2,9,30,3,1,2],[1,1,4,3,3,8,2,33,2],[1,2,3,10,14,30,3,3],[2,12,11,6,30,2,4],[2,13,11,6,30,2,5],[22,21,30,8]]
    #runs_col = [[6,1,2,1,4,7],[1,1,2,1,2,2,3,7],[1,1,2,1,5,2,1,5,7],[1,1,2,1,2,3,1,7,7],[6,6,2,1,10,16],[1,3,1,5,1,17,3],[2,2,2,4,3,3,1],[5,2,2,5,4,3,1],[1,3,1,2,4,9,2,1],[3,1,1,2,6,9,5,11,1],[3,2,1,1,2,1,7,5,3,2,1],[7,2,1,1,1,1,2,1,17,1,1],[1,1,1,2,1,1,1,1,1,1,14,3],[1,1,2,3,1,1,1,18,10,3],[1,2,1,2,2,3,1,15,7,2,3],[1,1,1,2,1,8,21,1,3],[1,1,1,2,2,10,10,7,1,3],[1,1,3,2,6,2,1,8,7,5],[1,1,4,2,4,3,1,6,7,4],[1,5,2,1,4,4,7,4],[1,5,3,4,7,3],[1,5,3,1,1,7,3],[1,6,2,2,1,1,1,7,2],[7,1,2,1,1,1,7,2],[7,3,2,1,4,1,1,1,7,2],[8,8,2,3,2,1,1,1,7,1],[8,12,2,2,3,1,1,7,1],[9,12,1,1,4,15],[9,5,5,7,3,6],[10,4,1,2,9,3,3],[11,4,1,1,2,3,3,10,3],[11,4,1,2,1,2,3,1,18,2,3],[12,3,3,2,5,1,1,7,2,3],[12,12,3,3,2,3,1,7,6],[12,24,4,7,5],[14,2,8,3,2,11,5],[14,1,1,7,3,2,5,4],[15,1,2,2,4,2,3,1,4],[15,1,2,4,1,4,10,2,3],[15,1,2,1,6,9,1,3],[16,6,2,2,1,1,1,7,2,2,2],[13,1,2,2,3,2,1,1,1,7,5,2],[12,1,3,3,1,2,1,1,1,7,4,1],[11,1,5,5,1,1,1,1,1,7,3,1],[10,1,1,6,1,1,1,1,1,1,7,4,1],[9,2,1,4,1,1,1,1,1,1,15,1],[8,1,1,5,1,3,1,1,1,5,6],[7,1,1,6,1,1,1,1,2,5],[6,1,5,5,1,1,4,4],[5,2,3,3,7,1,3,4],[4,1,2,2,4,6,1,2],[3,1,6,3,3,2,1,2],[2,1,2,2,1,1],[1,1,1,1,2,1,1],[1,1,1,1,7,2],[1,25,1,9,2,2],[1,25,1,1,12,1,2],[1,1,1,2,1,1,2,2,1,3],[1,1,2,1,1,1,1,1,10],[2,1,1,7,2,1,1,1,2,2,1,7],[3,1,3,9,4,1,4,4,1,6],[4,1,3,11,9,1,2,6],[5,2,12,3,6,6,6],[6,12,2,8,7],[7,15,6,27],[8,15,4,27],[9,9,32],[10,6,2,2,2,24],[11,5,2,3,2,23],[12,4,1,4,1,22],[13,3,2,1,1,3,2,21],[15,2,2,1,4,1,21],[14,2,3,1,4,1,21],[14,2,1,1,1,4,1,21],[14,2,6,1,1,4,1,21],[14,3,2,5,1,4,1,21],[13,3,2,1,4,1,21],[12,5,2,1,1,3,2,22],[12,3,1,1,4,1,22],[12,3,2,3,2,24],[11,4,2,31],[1,11,4,33],[1,10,8,4,5,27],[1,9,13,4,9,7],[1,9,12,1,7,3,6],[1,8,2,15,4,4,5],[1,8,1,1,12,2,9],[1,1,7,2,10,6,1,6,8],[1,1,7,4,1,13,2],[1,1,6,4,1,3,2],[1,1,5,4,2,2,16],[1,1,1,5,4,1,2,3],[1,1,1,5,4,1,2,1,1,2],[1,1,1,4,4,5,2,1,1],[1,1,1,3,1,4,1,4,1],[1,1,1,2,7,2,1,2],[1,1,1,2,1,5,4,3],[1,1,1,1,4,1,7,4],[1,1,1,1,2,3,5,6],[1,1,1,1,7,11]]

    ## https://www.researchgate.net/publication/290264363_On_the_Difficulty_of_Nonograms
    ## Batenburg construction -> requires 120 sweeps
    five_strip1 = [(1,2,)+ 6*(1,), (2,) + 6 * (1,) + (2,), (1,), (2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    three_strip = [(1,), (18,), (1,1)]
    five_strip2 = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1),(1,),(2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    two_strip   = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1)]
    #runs_row = five_strip1 + three_strip + five_strip2 + three_strip + two_strip
    #runs_col = [(3,2,3,2,1),(1,1,2,1,1,2,1)] + [(1,)*7 for _ in range(15)] + [(2,1,3,1,3)]
    ## small but constraint propogation alone won't solve 
    #runs_row = [(4,),(4,),(1,),(1,1),(1,)] # not possible to solve with this solver
    #runs_col = [(1,),(2,1),(2,1),(2,1),(1,1)] # not possible to solve with this solver

    game = Nonogram(runs_row, runs_col)

    ## set solution
    #game.set_grid(solution)

    ## solve game
    start_time = time.time()
    #game.solve_brute()
    game.solve()
    #game.solve_fast()
    end_time = time.time()

    game.show_grid(show_instructions=True, symbols="x#.?")
    print(game.is_complete())
    print("time taken: {:.5f}s".format(end_time - start_time))

    plot_nonogram(game.grid)

    if 1==0:
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

                plot_nonogram(game.grid)

        print("time taken: {:.4f}s".format(time.time() - start_time))

    plt.show()