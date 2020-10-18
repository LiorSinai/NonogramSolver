"""
Lior Sinai, 17 October 2020

Mock regex

"""

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

def argsort(array, reverse=False):
    """ returns the indexes for the original array for values in the sorted array"""
    return  sorted(range(len(array)), key = lambda x: array[x], reverse=reverse)


def rank_guesses(grid, n_rows: int, n_cols: int):
    """ Give a heuristic ranking for guesses
    - 2-4 neighbor cells are solved https://webpbn.com/pbnsolve.html
    """
    rankings = []
    # 2-4 rankings
    for idx in range(n_cols * n_rows):
        i, j = idx // n_cols, idx % n_cols
        rank = 0
        if grid[i][j] == EITHER:
            neighbours = [(i + o1, j + o2) for o1, o2 in ((-1, 0), (+1, 0), (0, -1), (0, +1))]
            for i_n, j_n in neighbours:
                # increase rank if a neighbour is solved. Edges don't count as solved
                if (i_n >= 0 and i_n < n_rows) and (j_n >= 0 and j_n < n_cols) and (grid[i_n][j_n] != EITHER):
                    rank += 1
            if rank >= 2:
                rankings.append((rank, (i, j)))

    rankings.sort(reverse=True)
    return rankings


                    