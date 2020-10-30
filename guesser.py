"""
Lior Sinai, 17 October 2020

Mock regex

"""

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

from random import shuffle

def argsort(array, reverse=False):
    """ returns the indexes for the original array for values in the sorted array"""
    return  sorted(range(len(array)), key = lambda x: array[x], reverse=reverse)


def rank_solved_neighbours(grid):
    """ Give a heuristic ranking for guesses
    - 2-4 neighbor cells are solved https://webpbn.com/pbnsolve.html
    """
    n_rows, n_cols = len(grid), len(grid[0])
    rankings = []
    rank_max = 0
    # increase rank for each solved neighbour
    for idx in range(n_cols * n_rows):
        i, j = idx // n_cols, idx % n_cols
        rank = 0
        if grid[i][j] == EITHER:
            neighbours = [(i + o1, j + o2) for o1, o2 in ((-1, 0), (+1, 0), (0, -1), (0, +1))]
            for i_n, j_n in neighbours:
                if (i_n >= 0 and i_n < n_rows) and (j_n >= 0 and j_n < n_cols) and (grid[i_n][j_n] != EITHER):
                    rank += 1   # increase rank if a neighbour is solved. 
            # increase rank if on edge -> these count as "solved" cells
            rank += (i == 0 or i == n_rows - 1)
            rank += (j == 0 or j == n_cols - 1)
            rank_max = max(rank, rank_max)
                
            rankings.append((rank, (i, j)))

    rankings = [x for x in rankings if x[0] == rank_max]
    #rankings.sort(reverse=True, key=lambda x: x[0]) 
    #rankings = [x for x in rankings if x[0] >= 2]
    return rankings


def score_Wolter(line, runs):
    """ Jan Wolters adhoc scoring algorithm. Favours lines with lower slack (free whites) and fewer clues"""
    free_whites = len(line) - sum(runs) - line.count(WHITE)
    return 2*len(runs) + free_whites


def get_sequence(arr):
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


def rank_guesses2(grid, runs_row, runs_col):
    """ Give a heuristic ranking for guesses
    Cells next to the largest incomplete maximums
    """
    rankings = []
    n_rows, n_cols = len(grid), len(grid[0])

    for i in range(n_rows):
        row = grid[i]
        rankings_i = get_rankings_next_to_maxs(row, runs_row[i])
        for rank, j in rankings_i:
            neighbours = [(i + o1, j + o2) for o1, o2 in ((-1, 0), (+1, 0), (0, -1), (0, +1))]
            for i_n, j_n in neighbours:
                # increase rank if a neighbour is solved. Edges don't count as solved
                if (i_n >= 0 and i_n < n_rows) and (j_n >= 0 and j_n < n_cols) and (grid[i_n][j_n] != EITHER):
                    rank += 1
            rankings.append((rank, (i, j)))
    for j in range(n_cols):
        col = [grid[i][j] for i in range(n_rows)]
        rankings_i = get_rankings_next_to_maxs(col, runs_col[j])
        for rank, i in rankings_i:
            neighbours = [(i + o1, j + o2) for o1, o2 in ((-1, 0), (+1, 0), (0, -1), (0, +1))]
            for i_n, j_n in neighbours:
                # increase rank if a neighbour is solved. Edges don't count as solved
                if (i_n >= 0 and i_n < n_rows) and (j_n >= 0 and j_n < n_cols) and (grid[i_n][j_n] != EITHER):
                    rank += 1
            rankings.append((rank, (i, j)))
    
    rankings.sort(reverse=True, key=lambda x: x[0])
    return rankings     

def get_rankings_next_to_maxs(array, runs_target):
    runs, pos = get_sequence(array)
    min_length = min(len(runs), len(runs_target))
    targets_sorted = sorted(runs_target, reverse=True)
    placed_orded = argsort(runs, reverse=True)
    k = 0
    rankings = []
    while k < min_length and runs[placed_orded[k]] == targets_sorted[k]:
        k += 1
    if k >= min_length - 1:
        return rankings  # this array is complete
    idx = placed_orded[k]
    if runs[idx] < runs_target[k] and (runs[idx] > targets_sorted[k+1]): 
        # this run is incomplete and definitely not part of the next run
        inds = (max(pos[k][0] - 1, 0), min(pos[k][1] + 1, len(array) - 1))
        for j in inds:
            rank = 0
            if array[j] == EITHER:
                rankings.append((rank, j))
    return rankings
        
              