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


def find_match(array, pattern):
    return find_match_backwards(array, pattern)
    #return find_match_fowards(array, pattern)


def find_match_fowards(array, pattern, start=True):
    # forward algorithm -> goes very slowly if there is a black much further down, because finds all possibilities that can fit infront
    # supposed to mimic the regex expression "([32]*)([31]){x}([32]+)...([32]*)"
    # except returns the minumum match
    out = Match(pattern=pattern)
    if start: #([32]*) match zero or more white or either
        candidate = []
        n = 0
        while n < len(array) and (n==0 or array[n-1] & WHITE):
            out = find_match_fowards(array[n:], pattern, start=False)
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
                out = find_match_fowards(array[pattern[0]+n:], pattern[1:], start=False)
                if out.is_match:
                    out.match = candidate + out.match
                    break   
                n += 1
    return out


def find_match_backwards(array, pattern, start=True):
    # returns the left-most match, constructed from the back
    # jumps backwards from the last placed black, and tries to 
    out = Match(pattern=pattern)
    if not pattern:
        candidate = [WHITE] * len(array)
        if fits(candidate, array):
            return Match(candidate, pattern=pattern)
    elif not array:
        return out
    elif array[0] == WHITE:
        # strip whites at the beginning
        i = 0
        while array[i] == WHITE:
            i += 1
        candidate = [WHITE] * i
        out = find_match_fowards(array[i:], pattern)
        if out.is_match:
            out.match = candidate + out.match 
    else:
        min_length = sum(pattern) + (len(pattern) - 1) 
        p = pattern[-1]
        min_left = 0 if len(pattern) == 1 else min_length - p - 1  # =sum(pattern[:-1]) + ((len(pattern)-1)-1)
        if min_left < -1:
            return out # not possible to match
        add_white = int(min_left > 0)
        
        candidate = []
        candidate += [EITHER] * min_left
        try: # find the last placed black
            # min_length-1 <= idx_start <= len(array) - p 
            idx_start =  min(listRightIndex(array, BLACK) - p + 1, len(array) - p)
            if idx_start > min_left:
                candidate += [WHITE] * (idx_start - min_left) # as many whites as possible
                candidate += [BLACK] * p
            else:
                raise ValueError # this black might belong to another sequence
        except ValueError:
            idx_start = min_left + add_white
            # make minimum sequence
            candidate += [WHITE] * add_white + [BLACK] * p
        candidate += [WHITE] * (len(array) - len(candidate)) # as many whites as needed at the end
        
        # first see if the whites need to change
        for idx in range(min_left, idx_start): 
            # find recursive matches
            if fits(candidate[idx:], array[idx:]):
                out = find_match_backwards(array[:idx], pattern[:-1])
                if out.is_match:
                    out.match = out.match + candidate[idx:]
                    break 
            candidate[idx] = EITHER # this can't be a white
        
        if not out.is_match: # this works by itself but is slow
            # shift pattern across and leave hard thinking for the next call
            for idx in range(min_left, len(array) - p + 1):
                candidate = []
                candidate += [EITHER] * (idx - add_white)
                candidate += [WHITE] * add_white
                candidate += [BLACK] * p
                candidate += [WHITE] * (len(array) - len(candidate)) # whites at the end 
                if fits(candidate, array):
                    out = find_match_backwards(array[:idx-add_white], pattern[:-1])
                    if out.is_match:
                        out.match = out.match + candidate[idx-add_white:]
                        break 
            
    return out


if __name__ == '__main__':    
    # for row, result in tests:
    #     s = ''.join(map(str, row))
    #     pattern = "([3]*)"
    #     for x in runs:
    #         pattern += "[31]{" + str(x) + "}([32]+)"
    #     pattern += "([3]*)"
    #     m1 = re.search(pattern,s)
    #     print(m1)

    runs = (6, 7)
    row = [WHITE] + [BLACK] *5 + [EITHER] * 2 + [BLACK] * 7
    result = [WHITE] + [BLACK] *6 + [WHITE] + [BLACK] * 7
    m2 = find_match(row, runs)
    print(m2.match == result, m2.match)
    print("")


    #http://scc-forge.lancaster.ac.uk/open/nonogram/ls-fast
    s = "---#--         -      # "
    sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
    reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}
    row = [sym_map[x] for x in s]
    run = (1, 1, 5)
    print("original:        ", s)
    left_most = find_match(row, run).match
    left_most = ''.join([reverse_map[x] for x in left_most])
    print("left-most match: ", left_most, left_most=="---#--#-----------#####-")
    right_most = find_match(row[::-1], run[::-1]).match[::-1]
    right_most = ''.join([reverse_map[x] for x in right_most])
    print("right-most match:", right_most, right_most=="---#-------------#-#####")