"""
Lior Sinai, 15 October 2020

Mock regex

"""

import re

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

def fits(a, b):
    return all(x & y for x, y in zip(a, b))


class Match():
    def __init__(self, match=None, pattern=None, span=None):
        self.match = match
        self.pattern = pattern
        self.span = span

    @property
    def is_match(self):
        return self.match is not None


def listRightIndex(array, value):
    """Returns the index for the right-most matching value"""
    return len(array) - array[-1::-1].index(value) -1


def minimum_sequence(pattern):
    """ Returns the minimum sequence of a pattern, which is one WHITE between every black"""
    arr = []
    for n_black in pattern:
        arr += [WHITE] + [BLACK]  * n_black
    return arr[1:]

def find_match(array, pattern, start=True):
    # supposed to mimic the regex expression "([32]*)([31]){x}([32]+)...([32]*)"
    # except returns the minumum match
    out = Match(pattern=pattern)
    if start: #([32]*) match zero or more white or either
        candidate = []
        n = 0
        while n < len(array) and (n==0 or array[n-1] & WHITE):
            out = find_match(array[n:], pattern, start=False)
            if out.is_match:
                out.match = candidate + out.match
                break
            candidate += [WHITE] # add another white
            n += 1
    elif not pattern:  # ([32]*) match end of array with zero or more whites
        candidate = [WHITE] * len(array)
        if fits(candidate, array):
            return Match(candidate, pattern=pattern)
    elif not array:
        return out
    else: # ([31]){x}([32]+) match the pattern object
        candidate = [BLACK] * pattern[0]
        if fits(candidate, array): 
            if len(array) == pattern[0] and len(pattern) == 1: # at end of array
                return Match(candidate, pattern)
            # add at least one white
            n = 1
            while (n < len(array) - pattern[0] + 1) and (array[pattern[0]+n-1] & WHITE): 
                candidate += [WHITE]
                out = find_match(array[pattern[0]+n:], pattern[1:], start=False)
                if out.is_match:
                    out.match = candidate + out.match
                    break   
                n += 1
    return out



if __name__ == '__main__':
    run = [3, 2, 1]

    tests = [
        ([3] * 10 , [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 + [WHITE] *  2),
        ([BLACK]*4 + [3]*6 , None), 
        ([BLACK]*3 + [WHITE]*5 + [BLACK] * 2 , None),
        ([WHITE] * 4 + [BLACK] *3 + [3] * 3, None),
        ([3, 3, BLACK, 3, 3, 3, BLACK, BLACK, 3, BLACK], 
         [BLACK, BLACK, BLACK, WHITE, WHITE, WHITE, BLACK, BLACK, WHITE, BLACK]),
         ([3, 3, 3, 3, BLACK, 3, BLACK, BLACK, 3, BLACK], 
         [WHITE, WHITE, BLACK, BLACK, BLACK, WHITE, BLACK, BLACK, WHITE, BLACK])
    ]
    
    

    for row, result in tests:
        s = ''.join(map(str, row))
        pattern = "([3]*)"
        for x in run:
            pattern += "[31]{" + str(x) + "}([32]+)"
        pattern += "([3]*)"
        m1 = re.search(pattern,s)
        print(m1)

    for row, result in tests:
        m2 = find_match(row, run, start=True)
        print(m2.match == result, m2.match)


    # right most
    row = [3] * 10
    result = [WHITE] *  2 + [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 
    m2 = find_match(row[::-1], run[::-1], start=True)
    print(m2.match[::-1] == result, m2.match[::-1] )

    row = [3, 3, 3, 3, 3, 3, 3, BLACK, 3, 3]
    result = [WHITE] *  2 + [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 
    m2 = find_match(row[::-1], run[::-1], start=True)
    print(m2.match[::-1] == result, m2.match[::-1] )
    m2 = find_match(row, run, start=True)
    print("----", m2.match)

    #http://scc-forge.lancaster.ac.uk/open/nonogram/ls-fast
    s = "---#--         -      # "
    sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
    reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}
    row = [sym_map[x] for x in s]
    run = (1, 1, 5)
    print(s)
    left_most = find_match(row, run).match
    print(''.join([reverse_map[x] for x in left_most]))
    right_most = find_match(row[::-1], run[::-1]).match[::-1]
    print(''.join([reverse_map[x] for x in right_most]))