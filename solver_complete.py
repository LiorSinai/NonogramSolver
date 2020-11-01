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

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

import matplotlib.pyplot as plt
import time

from nonogram import Nonogram, plot_nonogram, update_nonogram_plot
from integerPartitions import unique_perm_partitions


def solve(nonogram_):
    "Do constraint propagation before exhaustive search. Based on the RosettaCode code"
    exhaustive_search_max_iters = 1e6

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
        fixed = grid[i]
        possible_rows[i] = [row for row in possible_rows[i] if fits(fixed, row)] # reduce the amount of possible rows
        allowed = find_allowed(possible_rows[i])
        for j in range(n_cols):
            if fixed[j] != allowed[j]:
                columns_to_edit.add(j)
                grid[i]= allowed

    def fix_col(j):
        fixed = [grid[i][j] for i in range(n_rows)]
        possible_cols[j] = [col for col in possible_cols[j] if fits(fixed, col)] # reduce the amount of possible cols
        allowed = find_allowed(possible_cols[j])
        for i in range(n_rows):
            if fixed[i] != allowed[i]:
                rows_to_edit.add(i)
                grid[i][j] = allowed[i]

    # extract values from Nonogram object
    n_rows, n_cols = nonogram_.n_rows, nonogram_.n_cols
    runs_row, runs_col = nonogram_.runs_row, nonogram_.runs_col
    grid = [row[:] for row in nonogram_.grid]

    # initialise plot
    save, instruct, plot_progess =True, True, False # seriously slows down code
    if plot_progess:
        ax = plot_nonogram(grid, save=save, filename="solving_sweep_0", show_instructions=instruct, runs_row=runs_row, runs_col=runs_col)
    else:
        ax=None
    
    # initialise rows and columns. This reduces the amount of valid configurations for generate_sequences
    for i in range(n_rows):
        grid[i] = initialise(n_cols, runs_row[i])
    for j in range(n_cols):
        col = initialise(n_rows, runs_col[j])
        for i in range(n_rows):
            grid[i][j] = col[i]
    update_nonogram_plot(grid, ax=ax, save=save, filename="solving_sweep_2", plot_progess=plot_progess)

    # generate ALL possible sequences. SLOW and MEMORY INTENSIVE
    possible_rows = [generate_sequences(grid[i], runs_row[i]) for i in range(n_rows)]
    possible_cols = []
    for j in range(n_cols):
        col = [grid[i][j] for i in range(n_rows)]
        possible_cols.append(generate_sequences(col, runs_col[j]))

    print("initial")
    n_possible_rows = [len(x) for x in possible_rows]
    n_possible_cols = [len(x) for x in possible_cols]
    print("possible rows: {}\npossible columns: {}".format(n_possible_rows, n_possible_cols))
    print("summary: {} possible rows and {} possible columns".format(sum(n_possible_rows), sum(n_possible_cols)))

    # rows, columns for constraint propagation to be applied
    rows_to_edit = set(range(n_rows))
    columns_to_edit = set(range(n_cols))

    sweeps = 2 # include initialising
    while columns_to_edit:
        # constraint propagation
        for j in columns_to_edit:
            fix_col(j)
        sweeps += 1
        update_nonogram_plot(grid, ax=ax, save=save, filename="solving_sweep_{}".format(sweeps), plot_progess=plot_progess)
        columns_to_edit = set()
        for i in rows_to_edit:
            fix_row(i)
        sweeps += 1
        update_nonogram_plot(grid, ax=ax, save=save, filename="solving_sweep_{}".format(sweeps), plot_progess=plot_progess)
        
        rows_to_edit = set()
        
    print("\nconstraint propagation done in {} rounds".format(sweeps))
    n_possible_rows = [len(x) for x in possible_rows]
    n_possible_cols = [len(x) for x in possible_cols]
    print("possible rows: {}\npossible columns: {}".format(n_possible_rows, n_possible_cols))
    print("summary: {} possible rows and {} possible columns".format(sum(n_possible_rows), sum(n_possible_cols)))
    possible_combinations = 1
    for x in n_possible_rows:
        possible_combinations *= x
    print("         {:e} possibile combinations".format(possible_combinations))
    print("")

    solution_found = all(grid[i][j] in (BLACK, WHITE) for j in range(n_cols) for i in range(n_rows)) # might be incorrect
    if solution_found:
        print("Solution is unique") # but could be incorrect!
    elif possible_combinations >= exhaustive_search_max_iters:            
        print("Not trying exhaustive search. Too many possibilities")
    else:
        print("Solution may not be unique, doing exhaustive search:")
    
    def try_all(grid, i = 0) :
        if i >= n_rows:
            for j in range(n_cols):
                col = [row[j] for row in grid]
                if col not in possible_cols[j]:
                    return 0
            nonogram_.show_grid(grid)
            print("")
            return 1
        sol = 0
        for row in possible_rows[i]:
            grid[i] = row
            if nonogram_.is_valid_partial_columns(grid):
                grid_next = [row[:] for row in grid]
                sol += try_all(grid_next, i + 1)    
        return sol
    
    # start exhaustive search if not solved
    if not solution_found and possible_combinations < exhaustive_search_max_iters:
        grid_next = [row[:] for row in grid]
        num_solutions = try_all(grid_next, i=0)
        if num_solutions == 0:
            print("No solutions found")
        elif num_solutions == 1:
            print("Unique solution found")
        else: # num_solutions > 1:
                print("{} solutions found".format(num_solutions))

    return grid


if __name__ == '__main__':
    # the wikipedia W
    r_row = [(8, 7, 5, 7),(5, 4, 3, 3),(3, 3, 2, 3),(4, 3, 2, 2),(3, 3, 2, 2),(3, 4, 2, 2),(4, 5, 2),(3, 5, 1),(4, 3, 2),(3, 4, 2),(4, 4, 2),(3, 6, 2),(3, 2, 3, 1),(4, 3, 4, 2),(3, 2, 3, 2),(6, 5),(4, 5),(3, 3),(3,3), (1, 1)]
    r_col = [(1,),(1,),(2,),(4,),(7,),(9,),(2, 8),(1, 8),(8,),(1, 9),(2, 7),(3, 4),(6, 4),(8, 5),(1, 11),(1, 7),(8,),(1, 4, 8),(6, 8),(4, 7),(2, 4),(1, 4),(5,),(1, 4),(1,5),(7,),(5,),(3,),(1,),(1,)]

    # elephant
    #r_row = [(3,),(4,2),(6,6),(6,2,1),(1,4,2,1), (6,3,2),(6,7),(6,8),(1,10),(1,10), (1,10),(1,1,4,4),(3,4,4),(4,4),(4,4)]
    #r_col = [(1,),(11,),(3,3,1),(7,2),(7,), (15,), (1,5,7),(2,8),(14,),(9,), (1,6),(1,9),(1,9),(1,10),(12,)]

    # # ## chess board, multiple solutions
    #r_row = [(1,), (1,), (1,)]
    #r_col = [(1,), (1,), (1,)]

    # # Bonus
    #r_row = [(3,),(1,),(1,),(1,1),(1,),(1,1,4,1),(2,1,1,1,4),(1,1,1,1,1,1),(1,1,1,1,1),(2,1,1,1,1),(1,4,2,1,1,1),(1,3,1,4,1)]
    #r_col = [(0,),(1,1,1),(1,5),(7,1),(1,),(2,),(1,),(1,),(1,),(0,),(2,),(1,6),(0,),(6,),(1,1),(1,1),(1,1),(6,),(0,),(1,),(7,),(1,),(1,),(1,),(0,)]

    # # aeroplane -> solve fast doesn't work. https://www.youtube.com/watch?v=MZQDDzzRBvI
    #r_col = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,2,1],[3,3],[3,4],[2,1,2]]
    #r_row = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,4],[3,1,2],[3,2,2],[2,1,1]]

    ## https://www.researchgate.net/publication/290264363_On_the_Difficulty_of_Nonograms
    ## Batenburg construction -> requires 120 sweeps
    five_strip1 = [(1,2,)+ 6*(1,), (2,) + 6 * (1,) + (2,), (1,), (2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    three_strip = [(1,), (18,), (1,1)]
    five_strip2 = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1),(1,),(2,1,1,2,1,2,1),(1,1,1,2,1,1,1)]
    two_strip   = [(1,2,1,2,1,1,2),(1,1,1,2,1,1,1)]
    #r_row = five_strip1 + three_strip + five_strip2 + three_strip + two_strip
    #r_col = [(3,2,3,2,1),(1,1,2,1,1,2,1)] + [(1,)*7 for _ in range(15)] + [(2,1,3,1,3)]
    ## small but constraint propagation alone won't solve 
    #r_row = [(4,),(4,),(1,),(1,1),(1,)] # not possible to solve with this solver
    #r_col = [(1,),(2,1),(2,1),(2,1),(1,1)] # not possible to solve with this solver

    puzzle = Nonogram(r_row, r_col)

    ## set solution
    #puzzle.set_grid(solution)

    ## solve game
    start_time = time.time()
    grid = solve(puzzle)
    puzzle.set_grid(grid)
    end_time = time.time()

    puzzle.show_grid(show_instructions=True, to_file=False, symbols="x#.?")

    print(puzzle.is_complete(), "{:.2f}%%".format(puzzle.progress*100))
    print("time taken: {:.5f}s".format(end_time - start_time))
    print("solved with {} guesses".format(puzzle.guesses))

    plot_nonogram(puzzle.grid)

    if 1==1:
        start_time = time.time()
        #filename = 'rosetta_code_puzzles.txt'
        filename = "activity_workshop_puzzles.txt"  ##  https://activityworkshop.net/puzzlesgames/nonograms 
        with open(filename) as file:
            lines = file.read().split("\n")
            for i in range(1, len(lines), 3):
                s = lines[i + 1].strip().split(" ")
                r_row = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]
                s = lines[i + 2].strip().split(" ")
                r_col = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]

                puzzle = Nonogram(r_row, r_col)
                grid = solve(puzzle)
                puzzle.set_grid(grid)
                puzzle.show_grid(show_instructions=True, to_file=False, symbols="x#.?")
                print(puzzle.is_complete(), "{:.2f}%%".format(puzzle.progress*100))
                print("")

                plot_nonogram(puzzle.grid)

        print("time taken: {:.4f}s".format(time.time() - start_time))

    plt.show()