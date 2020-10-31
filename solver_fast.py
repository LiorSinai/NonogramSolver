"""
Lior Sinai, 9 October 2020

Solves a nonogram puzzles using only logic and constraint propagation. Might not solve but it';'s much faster than solve_complete.

See https://en.wikipedia.org/wiki/Nonogram

Based off the codililty coding question in nonogram.py, but optimsied and generalised

Notation
- # black
- . space/white
- ? unknown 

Several methods
- left-most right-most overlap
- guessing

"""

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

import matplotlib.pyplot as plt
import time
from copy import copy

from nonogram import Nonogram, plot_nonogram, update_nonogram_plot
#from matcher import Match, find_match
from matcher_nfa import Match, find_match, NonDeterministicFiniteAutomation
from guesser import rank_solved_neighbours, rank_guesses2

class SolvingError(Exception):
    pass

def solve_fast_(grid, nonogram_, rows_to_edit=None, columns_to_edit=None, make_guess=False):
    "Solve using logic and constraint propagation. Might not work"

    # important: pass one Nonogram object around but a separate grid object is edited on each recursive call

    def simple_filler(arr, runs):
        """ fill in black gaps and whites ending sequences. The overlap algorithm might miss these"""
        k = 0  # index for runs
        on_black = 0
        allowed = arr[:]
        for i in range(len(arr)):
            if arr[i] == WHITE:
                if on_black > 0:
                    k += 1 # move to the next pattern
                on_black = 0
            elif arr[i] == BLACK:
                on_black += 1
            else: # arr[i] == EITHER
                if k >= len(runs):
                    break
                elif (0 < on_black < runs[k]): # this must be part of this sequence
                    allowed[i] = BLACK
                    on_black += 1
                elif on_black == 0 and k > 0 and i >0 and arr[i-1] == BLACK: # this must be a white ending a sequence
                    allowed[i] = WHITE      
                else:
                    break # too many unknowns
        
        # put whites next to any 1 runs. Very special case
        if all([r == 1 for r in runs]):
            for i in range(len(arr)):
                if arr[i] == BLACK:
                    if i > 0:
                        allowed[i-1] = WHITE
                    if i < len(arr) - 1:
                        allowed[i+1] = WHITE

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
        left = nonogram_.NFA.find_match(arr, runs)
        right = nonogram_.NFA.find_match(arr[::-1], runs[::-1])
        if left.is_match and right.is_match:
            allowed = overlap(left.match, right.match[::-1])
        else:
            raise SolvingError("Left or right most match not found. A mistake was made")
        return allowed


    def splitter(arr, runs):
        """split rows at the max element. Then the strategies can be applied to each division. 
        This helps more with speed than with solving, because it speeds up the matching algorithm."""
        if not arr or not runs:
            return [(arr, runs)]
        runs_, positions = nonogram_._get_sequence(arr)
        split_value = max(runs)
        split_idx = runs.index(split_value)
        if runs.count(split_value) == 1 and split_value in runs_:
                idx = runs_.index(split_value)
                i0, i1 = positions[idx]
                i0, i1 = max(i0-1, 0), min(i1 + 1, len(arr)) # add whites on either side
                return splitter(arr[:i0], runs[:split_idx]) + [(arr[i0:i1+1], (runs[split_idx],))] + splitter(arr[i1+1:], runs[split_idx+1:])
        else:
            return [(arr, runs)]
                    
    def apply_strategies(array, runs):
        # allowed_full = [EITHER] * len(array)
        # allowed_full = [x & y for x, y in zip(allowed_full, left_rightmost_overlap(tuple(array), tuple(runs)))]
        # allowed_full = [x & y for x, y in zip(allowed_full, simple_filler(array, runs))]
        # allowed_full = [x & y for x, y in zip(allowed_full, simple_filler(array[::-1], runs[::-1])[::-1])]
        allowed_full = []
        for split in splitter(array, runs):
            segment, runs_segment = split
            if not segment:
                continue
            allowed = [EITHER] * len(segment)
            allowed = [x & y for x, y in zip(allowed, left_rightmost_overlap(tuple(segment), tuple(runs_segment)))]
            allowed = [x & y for x, y in zip(allowed, simple_filler(segment, runs_segment))]
            allowed = [x & y for x, y in zip(allowed, simple_filler(segment[::-1], runs_segment[::-1])[::-1])] # going from right
            allowed_full.extend(allowed)
        return allowed_full

    def fix_row(i):
        row = grid[i]
        allowed = apply_strategies(row, runs_row[i])
        for j in range(n_cols):
            if row[j] != allowed[j] and allowed[j]!=EITHER:
                columns_to_edit.add(j)
                grid[i]= allowed
    
    def fix_col(j):
        col = [grid[i][j] for i in range(n_rows)]
        allowed = apply_strategies(col, runs_col[j])
        for i in range(n_rows):
            if col[i] != allowed[i] and allowed[i]!=EITHER:
                rows_to_edit.add(i)
                grid[i][j] = allowed[i]
    
    # extract values from Nonogram object
    n_rows, n_cols = nonogram_.n_rows, nonogram_.n_cols
    runs_row, runs_col = nonogram_.runs_row, nonogram_.runs_col

    # initialise plot
    save, instruct, plot_progess =True, True, False # seriously slows down code
    if plot_progess:
        ax = plot_nonogram(grid, save=save, filename="solving_sweep_0", show_instructions=instruct, runs_row=runs_row, runs_col=runs_col)
    else:
        ax=None
    
    if rows_to_edit is None and columns_to_edit is None:
        # initialise
        # rows, columns for constraint propagation to be applied
        rows_to_edit = set()
        columns_to_edit = set(range(n_cols)) 

        for i in range(n_rows): 
            fix_row(i)
        sweeps = 1 # include initialise
    else:
        sweeps = 0  

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
    if nonogram_.guesses == 0:
        print("constraint propagation done in {} sweeps".format(sweeps))

    def probe(grid):
        """ solve every guess find the guess which makes the most progress on the next guess"""
        rankings = rank_solved_neighbours(grid)
        max_solve = 0
        guess = None
        for rank, ij in rankings: # only probe the top 10
            i, j = ij
            for value in [BLACK, WHITE]:
                grid_next = [row[:] for row in grid]
                grid_next[i][j] = value
                try:
                    grid_next = solve_fast_(grid_next, nonogram_, {i}, {j}, make_guess=False)
                    progress = 1 - sum(row.count(EITHER) for row in grid_next)/(n_rows * n_cols)
                except SolvingError:
                    progress = 0
                    return i,j, [value ^ 3], rank, progress # the other value is definitely not a guess
                if progress >= 1:
                    return i,j, [value], rank, progress # solution found
                elif progress > max_solve:
                    max_solve = progress
                    guess = i,j, [BLACK, WHITE], rank, progress  # got stuck, either might be correct
        return guess


    if not nonogram_.is_complete(grid) and make_guess:
        # rankings = rank_solved_neighbours(grid)
        # rank, ij = rankings[0] # only make a guess with the highest ranked 
        # i, j = ij
        # values = [BLACK, WHITE]

        guess =  probe(grid) 
        if guess is None: 
            raise SolvingError("all guesses from this configuration are are wrong")
        i,j, values, rank, prog = guess

        progress = 1 - sum(row.count(EITHER) for row in grid)/(n_rows * n_cols)
        is_guess = " guess" if len(values) > 1 else ""
        print(nonogram_.guesses, "{:.5f}%".format(progress*100), is_guess)

        # make a guess
        nonogram_.guesses += 1 # only the first one is a guess, the second time we know it is right
        for cell in values:
            grid_next = [row[:] for row in grid]
            grid_next[i][j] = cell
            try:
                grid_next = solve_fast_(grid_next, nonogram_, {i}, {j}, make_guess=True)
                if nonogram_.is_complete(grid_next):
                    grid = grid_next
                    break   
            except SolvingError:
                pass

    return grid

def solve_fast(nonogram_, make_guess=False):
    grid = [row[:] for row in nonogram_.grid]
    print("solving ...")
    nonogram_.NFA = NonDeterministicFiniteAutomation()
    grid = solve_fast_(grid, nonogram_, make_guess=make_guess)

    # print LRU cache info
    hits = nonogram_.NFA.find_match.cache_info().hits
    misses = nonogram_.NFA.find_match.cache_info().misses
    print("Cache: hits/misses={:.2f}".format(hits/misses), nonogram_.NFA.find_match.cache_info())

    return grid



if __name__ == '__main__':
    # the wikipedia W
    r_row = [(8, 7, 5, 7),(5, 4, 3, 3),(3, 3, 2, 3),(4, 3, 2, 2),(3, 3, 2, 2),(3, 4, 2, 2),(4, 5, 2),(3, 5, 1),(4, 3, 2),(3, 4, 2),(4, 4, 2),(3, 6, 2),(3, 2, 3, 1),(4, 3, 4, 2),(3, 2, 3, 2),(6, 5),(4, 5),(3, 3),(3,3), (1, 1)]
    r_col = [(1,),(1,),(2,),(4,),(7,),(9,),(2, 8),(1, 8),(8,),(1, 9),(2, 7),(3, 4),(6, 4),(8, 5),(1, 11),(1, 7),(8,),(1, 4, 8),(6, 8),(4, 7),(2, 4),(1, 4),(5,),(1, 4),(1,5),(7,),(5,),(3,),(1,),(1,)]

    # elephant
    r_row = [(3,),(4,2),(6,6),(6,2,1),(1,4,2,1), (6,3,2),(6,7),(6,8),(1,10),(1,10), (1,10),(1,1,4,4),(3,4,4),(4,4),(4,4)]
    r_col = [(1,),(11,),(3,3,1),(7,2),(7,), (15,), (1,5,7),(2,8),(14,),(9,), (1,6),(1,9),(1,9),(1,10),(12,)]

    # # ## chess board, multiple solutions
    #r_row = [(1,), (1,), (1,)]
    #r_col = [(1,), (1,), (1,)]

    # # Bonus
    #r_row = [(3,),(1,),(1,),(1,1),(1,),(1,1,4,1),(2,1,1,1,4),(1,1,1,1,1,1),(1,1,1,1,1),(2,1,1,1,1),(1,4,2,1,1,1),(1,3,1,4,1)]
    #r_col = [(0,),(1,1,1),(1,5),(7,1),(1,),(2,),(1,),(1,),(1,),(0,),(2,),(1,6),(0,),(6,),(1,1),(1,1),(1,1),(6,),(0,),(1,),(7,),(1,),(1,),(1,),(0,)]

    # # aeroplane -> solve fast doesn't work. https://www.youtube.com/watch?v=MZQDDzzRBvI
    r_col = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,2,1],[3,3],[3,4],[2,1,2]]
    r_row = [[2,2],[3,4],[3,6],[3,7],[3,5],[3,3],[1,4],[2,3],[8],[4,3],[4,6],[4,4],[3,1,2],[3,2,2],[2,1,1]]

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

    # solve game
    start_time = time.time()
    grid = solve_fast(puzzle, make_guess=True)
    puzzle.set_grid(grid)
    end_time = time.time()

    puzzle.show_grid(show_instructions=False, to_file=False, symbols="x#.?")

    print(puzzle.is_complete(), "{:.2f}%".format(puzzle.progress*100))
    print("time taken: {:.5f}s".format(end_time - start_time))
    print("solved with {} guesses".format(puzzle.guesses))
    print("")

    plot_nonogram(puzzle.grid, show_instructions=True, save=False, runs_row=r_row, runs_col=r_col)

    if 1==1:
        start_time = time.time()
        #filename = 'rosetta_code_puzzles.txt'
        filename = "activity_workshop_puzzles.txt"  ##  https://activityworkshop.net/puzzlesgames/nonograms 
        with open(filename) as file:
            lines = file.read().split("\n")
            for i in range(0, len(lines), 3):
                s = lines[i + 1].strip().split(" ")
                r_row = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]
                s = lines[i + 2].strip().split(" ")
                r_col = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]

                puzzle = Nonogram(r_row, r_col)
                grid = solve_fast(puzzle)
                puzzle.set_grid(grid)
                puzzle.show_grid(show_instructions=False, to_file=False, symbols="x#.?")
                print(puzzle.is_complete(), "{:.2f}%%".format(puzzle.progress*100))
                print("")

                plot_nonogram(puzzle.grid, show_instructions=True, save=False, runs_row=r_row, runs_col=r_col)

        print("time taken: {:.4f}s".format(time.time() - start_time))

    plt.show()