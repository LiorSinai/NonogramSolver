"""
Lior Sinai, 9 October 2020

Nonogram puzzle class. Inspired by a CodeSignal question.

See https://en.wikipedia.org/wiki/Nonogram


Notation
- # black
- . space/white
- ? unknown 

2 solutions methods
- constraint propagation
- exhaustive search 

"""

import matplotlib.pyplot as plt
import matplotlib.colors


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
        self.guesses = 0

    @property
    def num_solutions(self):
        return len(self.solutions)

    @property
    def progress(self):
        "the fraction of cells completed (correct or not)"
        return 1 - sum(row.count(EITHER) for row in self.grid)/(self.n_rows * self.n_cols)

    def get_col(self, grid, j):
        return [grid[i][j] for i in range(len(grid))]

    def set_grid(self, grid):
        self.grid = [row[:] for row in grid]

    def make_box(self, grid=None, symbols="x#.?"):
        grid = self.grid if grid is None else grid

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
        for i, row in enumerate(grid):
            run = self.runs_row[i]
            row = ["-"] * (row_max_length - len(run)) + list(map(str, run)) + [symbols[x] for x in row]
            box.append(row)
        
        return box

    def show_grid(self, grid=None, show_instructions=True, to_screen=True, to_file=False, symbols="x#.?"):
        if to_file:
            file = open("nonogram_grid.txt", "w")

        grid = self.grid if grid is None else grid
        if show_instructions:
            grid_ = self.make_box(grid, symbols) 
        else:
            grid_ = []
            for row in grid:
                grid_.append([symbols[x] for x in row]) # x-> something is wrong, #->black, .->white, ?->was never assigned
        for row in grid_:
            s = ''
            for symbol in row:
                if show_instructions:
                    s += "{:3}".format(symbol) # more room for the numbers
                else:
                    s += "{:2}".format(symbol)
            if to_screen:
                print(s)
            if to_file:
                file.write("\n"+s)

    def is_complete(self, grid=None):
        grid = self.grid if grid is None else grid

        for arr, run in zip(grid + list(zip(*grid)), self.runs_row + self.runs_col):
            if not self.is_valid_line(arr, run):
                return False    
        return True

    def is_valid_partial_grid(self, grid=None):
        if grid is None:
            grid = self.grid
        for arr, run in zip(grid + list(zip(*grid)), self.runs_row + self.runs_col):
            if not self.is_valid_partial(arr, run): # from left
                return False    
            if not self.is_valid_partial(arr[::-1], run[::-1]):
                return False # from right
        return True

    def is_valid_partial_columns(self, grid=None):
        if grid is None:
            grid = self.grid
        for arr, run in zip(list(zip(*grid)), self.runs_col):
            if not self.is_valid_partial(arr, run): # from top only
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
        if not sequence: # can match either a zero or nothing
            return list(target) == [] or list(target) == [0]
        return sequence == list(target)

    def is_valid_partial(self, arr, target):   
        "do placed black squares follow the rules so far?"

        try:
            idx = arr.index(EITHER) # make sure there are no gaps
            sequence, _ = self._get_sequence(arr[:idx])
            return all([x == y for x, y in zip(sequence, target)])
        except ValueError:
            return True # not sure if valid or not so return


    
def encode_puzzle(filename, runs_row, runs_col, description="") -> None:
    "Encode a puzzle in a text file in strings of letters"
    with open(filename, "a+") as file:
        file.write("\n--" + description)
        for runs in (runs_row, runs_col):
            s = "\n" + " ".join([''.join(map(lambda num: chr(num + ord('A') - 1), seq)) for seq in runs])
            file.write(s)


def plot_nonogram(grid, ax=None, save=False, filename="nonogoram_grid", 
                 show_instructions=False, runs_row=None, runs_col=None):
    n_large_puzzle  = 50
    n_medium_puzzle = 30

    if not ax:
        _, ax = plt.subplots(figsize=(12, 9))
    n_rows, n_cols = len(grid), len(grid[0])

    # custom color map
    cmap = matplotlib.colors.ListedColormap(['red', 'black', 'white', 'cornflowerblue'])
    boundaries = [0, 0.1, BLACK+0.01, WHITE+0.01, EITHER]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # plot
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

    # set grid lines. Majors must not overlap with minors
    ax.set_xticks([x-0.5 for x in list(range(1, n_cols + 1))], minor=True)
    ax.set_yticks([x-0.5 for x in list(range(1, n_rows + 1))], minor=True)
    plt.grid(which="minor", linewidth=1.1, color="k", alpha=0.7)

    if show_instructions:
        if runs_row is None or runs_col is None:
            raise UserWarning("runs_row and runs_col must not be None if show_instructions=True")
        ax.set_xticks(list(range(0, n_cols)), minor=False)
        ax.set_yticks(list(range(0, n_rows)), minor=False) 
        n = max(n_rows, n_cols)
        if n > n_large_puzzle:
            fontsize = 'xx-small'
        elif n > n_medium_puzzle:
            fontsize = 'medium'
        else:
            fontsize = 'large'
        ax.set_yticklabels([','.join(map(str,r)) for r in runs_row],  fontdict={'fontsize': fontsize})
        ax.set_xticklabels(['\n'.join(map(str,r)) for r in runs_col], fontdict={'fontsize': fontsize})

        #Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.tight_layout() # change layout so all the vertical ticks are in view. SLOW
    else:
        # set major ticks at reasonable intervals
        step_major = 5
        if min(n_rows, n_rows) // step_major < step_major:
                step_major = 1
        else:
            while min(n_rows, n_rows) // step_major > step_major:
                step_major += step_major
        ax.set_xticks(list(range(0, n_cols, step_major)), minor=False)
        ax.set_yticks(list(range(0, n_rows, step_major)), minor=False) 

    if save:
        fig = ax.figure
        fig.savefig(filename)
    return ax


def update_nonogram_plot(grid, ax, save=False, filename="nonogoram_grid", plot_progess=True):
    if not plot_progess:
        return
    ax.images[0].set_data(grid)
    if save:
        fig = ax.figure
        fig.savefig(filename)
