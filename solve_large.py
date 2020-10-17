"""
Lior Sinai, 15 October 2020

Transcibe for using in this solver: http://scc-forge.lancaster.ac.uk/open/nonogram/auto

"""

import time

import matplotlib.pyplot as plt
from nonogramSolver import Nonogram, plot_nonogram


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
            elif writing:
                active_runs.append(tuple(map(int, line.split(','))))

    return runs_row, runs_col


if __name__ == '__main__':
    #file_name = "lost_puzzle.txt"
    file_name = "beach_puzzle.txt" # can't solve
    #file_name = "artist_puzzle.txt"
    #file_name = "balance_puzzle.txt"
    #file_name = "warship_puzzle.txt"

    runs_row, runs_col = decode(file_name)

    game = Nonogram(runs_row, runs_col)

    ## solve game
    start_time = time.time()
    game.solve_fast()
    end_time = time.time()

    game.show_grid(show_instructions=False, symbols="x#.?") # these are too big to print on the command line
    print(game.is_complete(), "{:.2f}%%".format(game.progress*100))
    print("time taken: {:.5f}s".format(end_time - start_time))
    plot_nonogram(game.grid)

    plt.show()