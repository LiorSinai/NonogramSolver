"""
Lior Sinai, 15 October 2020

Transcibe for using in this solver: http://scc-forge.lancaster.ac.uk/open/nonogram/auto

"""

import time

import matplotlib.pyplot as plt
from nonogramSolver import Nonogram, plot_nonogram

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

def encode(runs_row, runs_col, filename="large_puzzle.txt"):
    max_rule = 0
    for runs in (runs_col, runs_row):
        for run in runs:
            max_rule = max(max_rule, max(run))

    with open(filename, "w+") as file:
        file.write("\ntitle \"large\"")
        file.write("\nby http://www.teall.info/2013/03/online-nonogram-solver.html")
        file.write("\n")
        file.write("\nmaxrule {}".format(max_rule))
        file.write("\nwidth {}".format(len(runs_col)))
        file.write("\nheight {}".format(len(runs_row)))
        file.write("\n")
        file.write("\nrows")
        for runs in (runs_row):
            file.write("\n" + ','.join(map(str, runs)))

        file.write("\n")
        file.write("\ncolumns")
        for runs in (runs_col):
            file.write("\n" + ','.join(map(str, runs)))

    
def decode(filename: str):
    runs_row = []
    runs_col = []
    solution = None

    writing = False
    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if line == '':
                break
            line = line.strip()
            if line == '':
                continue
            if line == 'rows':
                active_runs = runs_row
                writing = True
            elif line == 'columns':
                active_runs = runs_col
            elif line[:4] == 'goal':
                solution = line[6:-1]
            elif writing:
                line = line.strip(',')
                active_runs.append(tuple(map(int, line.split(','))))

    return runs_row, runs_col, solution


def decode_solution(solution, n_rows, n_cols):
    convertor = {"0": WHITE, "1": BLACK}

    solution = [solution[i * n_cols:i * n_cols +n_cols] for i in range(0, n_rows)]

    grid = []
    for row in solution:
        grid.append([convertor[c] for c in row])
    return grid

if __name__ == '__main__':
    #file_name = "lost_puzzle.txt"
    #file_name = "beach_puzzle.txt" # can't solve
    #file_name = "artist_puzzle.txt" # faster with match_forwards than match backwards
    #file_name = "balance_puzzle.txt"
    #file_name = "warship_puzzle.txt"
    file_name = "bear.txt"

    # the webpbn puzzles are super hard
    #file_name = "webpbn-00065-Mum's the Word.txt"

    file_name = 'puzzles/'+ file_name
    runs_row, runs_col, solution = decode(file_name)
    game = Nonogram(runs_row, runs_col)

    solve = True
    # set solution
    if solution and not solve: # the webpbn files have solutions
        print("setting solution ...")
        grid_sol = decode_solution(solution, len(runs_row), len(runs_col))
        game.set_grid(grid_sol)

    ##solve game
    if solve:
        start_time = time.time()
        game.solve_fast(make_guess=False)
        #game.solve()
        end_time = time.time()
        #game.show_grid(show_instructions=False, symbols="x#.?") # these are very big to print on the command line
        print(game.is_complete(), "{:.2f}%%".format(game.progress*100))
        print("time taken: {:.5f}s".format(end_time - start_time))

    plot_nonogram(game.grid)

    plt.show()