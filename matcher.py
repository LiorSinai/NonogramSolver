"""
Lior Sinai, 15 October 2020

Mock regex

"""

from Match import Match, minimum_sequence

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

def fits(a, b):
    return all(x & y for x, y in zip(a, b))


def listRightIndex(array, value):
    """Returns the index for the right-most matching value"""
    return len(array) - array[-1::-1].index(value) -1


def find_match(array, pattern):
    iters_threshold = 5e3
    return find_match_backwards(array, pattern, iters_threshold=iters_threshold)
    #return find_match_forwards(array, pattern, iters_threshold=iters_threshold)
    # try:
    #     threshold = 0.2 * len(array)
    #     idx = listRightIndex(array, BLACK)
    #     if idx < threshold:
    #         return find_match_forwards(array, pattern, iters_threshold=iters_threshold)
    #     else:
    #         return find_match_backwards(array, pattern, iters_threshold=iters_threshold)
    # except ValueError:
    #     return find_match_backwards(array, pattern, iters_threshold=iters_threshold)


def find_match_forwards(array, pattern, start=True, iters_threshold=1e9):
    def find_match_forwards_(array, pattern, start=True):
        # forward algorithm -> goes very slowly if there is a black much further down, because finds all possibilities that can fit infront
        # supposed to mimic the regex expression "([32]*)([31]){x}([32]+)...([32]*)"
        # except returns the minumum match
        nonlocal iters
        iters += 1
        out = Match(pattern=pattern)
        min_left = sum(pattern) + (len(pattern)-1)

        if iters > iters_threshold:
            return out

        if not pattern:  # ([32]*) match end of array with zero or more whites
            candidate = [WHITE] * len(array)
            if fits(candidate, array):
                return Match(candidate, pattern=pattern)
        elif not array:
            return out
        elif start: #([32]*) match zero or more white or either
            candidate = []
            i = 0
            # account for whites at the beginning
            while array[i] == WHITE and i < len(array):
                candidate += [WHITE]
                i += 1
            # find recursive matches
            while i <= (len(array) - min_left) and (i==0 or array[i-1] & WHITE):
                out = find_match_forwards_(array[i:], pattern, start=False)
                if out.is_match:
                    out.match = candidate + out.match
                    break
                candidate += [WHITE] # add another white
                i += 1
        else: # ([31]){x}([32]+) match the pattern object
            p = pattern[0]
            candidate = [BLACK] * p
            if fits(candidate, array): 
                if len(array) == p and len(pattern) == 1: # at end of array
                    return Match(candidate, pattern)
                # add at least one white
                i = 0
                while (i <= len(array) - min_left) and (array[p+i] & WHITE): 
                    candidate += [WHITE]
                    out = find_match_forwards_(array[p+i+1:], pattern[1:], start=False,)
                    if out.is_match:
                        out.match = candidate + out.match
                        break   
                    i += 1
        return out

    iters = 0
    iters_threshold = iters_threshold
    return find_match_forwards_(array, pattern, start=True)


def find_match_backwards(array, pattern, iters_threshold=1e9):
    # returns the left-most match, constructed from the back
    def find_match_backwards_(array, pattern):
        nonlocal iters
        iters += 1
        out = Match(pattern=pattern)
        
        if iters > iters_threshold:
            return out

        if not pattern:
            candidate = [WHITE] * len(array)
            if fits(candidate, array):
                return Match(candidate, pattern=pattern)
        elif not array:
            return out # no fit
        elif array[0] == WHITE:
            # strip whites at the beginning
            i = 0
            while i < len(array) and array[i] == WHITE:
                i += 1
            candidate = [WHITE] * i
            out = find_match_backwards_(array[i:], pattern)
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
            candidate += [WHITE] * add_white
            candidate += [BLACK] * p
            candidate += [WHITE] * (len(array) - len(candidate)) # whites at the end 

            fitter = [x & y for x, y in zip(candidate, array)]

            for idx in range(min_left+add_white, len(array) - p + 1):
                if idx - 1 >= 0:
                    fitter[idx - 1] = array[idx - 1] & WHITE
                if idx - add_white - 1 >= 0:
                    fitter[idx - add_white -1] = array[idx - add_white -1] & EITHER # this will overwrite the previous step if add_white=0
                fitter[idx] =  array[idx] & BLACK
                fitter[idx + p-1] = (array[idx + p - 1] & BLACK)
                fitter[idx] = fitter[idx] # dummy step to avoid werid optimisation errors
                
                #if fits(candidate[idx-add_white:], array[idx-add_white:]):
                if all(fitter[idx-add_white:]):
                    out = find_match_backwards_(array[:idx-add_white], pattern[:-1])
                    if out.is_match:
                        out.match = out.match + candidate[idx-add_white:]
                        break 
                # shift pattern across and leave hard thinking for the next call
                candidate = [EITHER] + candidate[:-1]
        return out
    
    iters = 0
    iters_threshold = iters_threshold
    return find_match_backwards_(array, pattern)

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