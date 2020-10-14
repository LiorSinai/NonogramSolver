"""
Rosetta Code Nonogram

https://rosettacode.org/wiki/Nonogram_solver#Python_3


"""

from functools import reduce
 
def gen_row(w, s):
    """Create all patterns of a row or col that match given runs."""
    def gen_seg(o, sp):
        if not o:
            return [[2] * sp]
        return [[2] * x + o[0] + tail
                for x in range(1, sp - len(o) + 2)
                for tail in gen_seg(o[1:], sp - x)]
 
    return [x[1:] for x in gen_seg([[1] * i for i in s], w + 1 - sum(s))]
 
 
def deduce(hr, vr):
    """Fix inevitable value of cells, and propagate."""
    def allowable(row):
        return reduce(lambda a, b: [x | y for x, y in zip(a, b)], row)
 
    def fits(a, b):
        return all(x & y for x, y in zip(a, b))
 
    def fix_col(n):
        """See if any value in a given column is fixed;
        if so, mark its corresponding row for future fixup."""
        c = [x[n] for x in can_do]
        cols[n] = [x for x in cols[n] if fits(x, c)]
        for i, x in enumerate(allowable(cols[n])):
            if x != can_do[i][n]:
                mod_rows.add(i)
                can_do[i][n] &= x
 
    def fix_row(n):
        """Ditto, for rows."""
        c = can_do[n]
        rows[n] = [x for x in rows[n] if fits(x, c)]
        for i, x in enumerate(allowable(rows[n])):
            if x != can_do[n][i]:
                mod_cols.add(i)
                can_do[n][i] &= x
 
    def show_gram(m):
        # If there's 'x', something is wrong.
        # If there's '?', needs more work.
        for x in m:
            print(" ".join("x#.?"[i] for i in x))
        print()
 
    w, h = len(vr), len(hr)
    rows = [gen_row(w, x) for x in hr]
    cols = [gen_row(h, x) for x in vr]
    can_do = list(map(allowable, rows))
 
    # Initially mark all columns for update.
    mod_rows, mod_cols = set(), set(range(w))
 
    while mod_cols:
        for i in mod_cols:
            fix_col(i)
        mod_cols = set()
        for i in mod_rows:
            fix_row(i)
        mod_rows = set()
 
    if all(can_do[i][j] in (1, 2) for j in range(w) for i in range(h)):
        print("Solution would be unique")  # but could be incorrect!
    else:
        print("Solution may not be unique, doing exhaustive search:")
 
    # We actually do exhaustive search anyway. Unique solution takes
    # no time in this phase anyway, but just in case there's no
    # solution (could happen?).
    out = [0] * h
 
    def try_all(n = 0):
        if n >= h:
            for j in range(w):
                if [x[j] for x in out] not in cols[j]:
                    return 0
            show_gram(out)
            return 1
        sol = 0
        for x in rows[n]:
            out[n] = x
            sol += try_all(n + 1)
        return sol
 
    n = try_all()
    if not n:
        print("No solution.")
    elif n == 1:
        print("Unique solution.")
    else:
        print(n, "solutions.")
    print()
 
 
def solve(s, show_runs=True):
    # s = [[[ord(c) - ord('A') + 1 for c in w] for w in l.split()]
    #      for l in p.splitlines()]
    if show_runs:
        print("Horizontal runs:", s[0])
        print("Vertical runs:", s[1])
    deduce(s[0], s[1])

if __name__ == '__main__':
    # the wikipedia W 
    runs_row = [(8, 7, 5, 7),(5, 4, 3, 3),(3, 3, 2, 3),(4, 3, 2, 2),(3, 3, 2, 2),(3, 4, 2, 2),(4, 5, 2),(3, 5, 1),(4, 3, 2),(3, 4, 2),(4, 4, 2),(3, 6, 2),(3, 2, 3, 1),(4, 3, 4, 2),(3, 2, 3, 2),(6, 5),(4, 5),(3, 3),(3,3), (1, 1)]
    runs_col = [(1,),(1,),(2,),(4,),(7,),(9,),(2, 8),(1, 8),(8,),(1, 9),(2, 7),(3, 4),(6, 4),(8, 5),(1, 11),(1, 7),(8,),(1, 4, 8),(6, 8),(4, 7),(2, 4),(1, 4),(5,),(1, 4),(1,5),(7,),(5,),(3,),(1,),(1,)]

    # elephant
    runs_row = [(3,),(4,2),(6,6),(6,2,1),(1,4,2,1), (6,3,2),(6,7),(6,8),(1,10),(1,10), (1,10),(1,1,4,4),(3,4,4),(4,4),(4,4)]
    runs_col = [(1,),(11,),(3,3,1),(7,2),(7,), (15,), (1,5,7),(2,8),(14,),(9,), (1,6),(1,9),(1,9),(1,10),(12,)]

    ## chess board, multiple solutions
    #runs_row = [(1,), (1,), (1,)]
    #runs_col = [(1,), (1,), (1,)]

    solve([runs_row, runs_col], show_runs=False)

    #filename = 'rosetta_code_puzzles.txt'
    filename = "activity_workshop_puzzles.txt"  ##  https://activityworkshop.net/puzzlesgames/nonograms 
    with open(filename) as file:
        lines = file.read().split("\n")
        for i in range(0, len(lines), 3):
            s = lines[i + 1].strip().split(" ")
            runs_row = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]
            s = lines[i + 2].strip().split(" ")
            runs_col = [tuple(ord(c) - ord('A') + 1 for c in run) for run in s]

            solve([runs_row, runs_col], show_runs=False)

 