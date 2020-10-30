BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

from match import Match, minimum_sequence, special_matches, listRightIndex


def can_place_block(line, idx, p):
    return idx + p <= len(line) and (idx+p == len(line) or line[idx+p] != BLACK) \
           and all([line[i] & BLACK for i in range(idx, idx+p)]) 

def is_free_space(line, start, end):
    return all([line[i] != BLACK for i in range(start, end + 1)])

def fits(a, b):
    return all(x & y for x, y in zip(a, b))


def shift_forwards(line, match, k, checkpoints, idx, pattern):
    k -= 1
    idx_prev = checkpoints[k]
    match_new = match[:checkpoints[k]]
    while ((idx >= idx_prev +  pattern[k]) or \
        not (can_place_block(line, idx_prev, pattern[k]))) and idx_prev + pattern[k] <= len(line):
        if line[idx_prev] & WHITE:
            match_new.insert(idx_prev, WHITE) 
            idx_prev += 1
            checkpoints[k] += 1
        elif k > 0: # backtrack again
            k -= 1
            idx_prev = checkpoints[k]
            match_new = match[:checkpoints[k]] 
        else:
            return [], -1, k
    return match_new, idx_prev, k


def find_match(line, pattern):
    # strategy: make forward match, then "push" wrong blocks to the back
    def find_forward_match():
        left = []
        checkpoints = []
        # make forward match
        idx, k = 0, 0
        while (idx < len(line) and k < len(pattern)):
            if (line[idx] == BLACK) or (line[idx] == EITHER):  # try place run
                if (can_place_block(line, idx, pattern[k])):
                    checkpoints.append(idx)
                    left += [BLACK] * pattern[k]
                    idx += pattern[k]
                    if idx < len(line):
                        left += [WHITE]
                        idx += 1
                    k += 1  # advance to next run
                elif line[idx] == EITHER:
                    idx += 1 # just move forward
                    left += [WHITE]
                elif k > 0: # try shift the last block forwards
                    left, idx, k = shift_forwards(line, left, k, checkpoints, idx, pattern)
                    if not left:
                        return [] # no match
                else:
                    return [] # no match
            else: # WHITE
                idx += 1 # just move forward
                left += [WHITE]
            if k == len(pattern):
                if line[idx:].count(BLACK) == 0:
                    return left
                else:
                    idx = line.index(BLACK, idx + 1)
                    left, idx, k = shift_forwards(line, left, k, checkpoints, idx, pattern)
                    
        if k != len(pattern):
            left = []
        return left
    
    def find_backward_match():
        right = []
        # make forward match
        idx, k = len(line) - 1, len(pattern) - 1
        while (idx > -1) and (k > -1):
            if (line[idx] == BLACK) or (line[idx] == EITHER):  # try place run
                if (idx - pattern[k] >= 0) and can_place_block(line, idx-pattern[k]+1, pattern[k]):
                    right = [BLACK] * pattern[k] + right
                    idx -= pattern[k]
                    if idx > 0:
                        right.insert(0, WHITE)
                        idx -= 1
                    k -= 1  # advance to next run
                    if k == -1:
                        break
                elif line[idx] == EITHER:
                    idx -= 1 # just move forward
                    right.insert(0, WHITE)
                else:
                    return right # no match
            else: # WHITE
                idx -= 1 # just move forward
                right.insert(0, WHITE)
        return right

    left  = find_forward_match()
    idx = len(left) - 1
    # try:
    #     right_most_idx = listRightIndex(line, BLACK)
    #     right = find_backward_match()
    # except ValueError: # no more blacks to place
    m = Match(pattern=pattern)
    if len(left) > 0: 
        m.match = left
        m.match +=  [WHITE] * (len(line) - len(left)) 
    
    return m
    
            


if __name__ == '__main__':
    matcher = find_match

    runs = (3, 2, 1)
    tests = [
        ([3] * 10 , [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 + [WHITE] *  2),
        ([BLACK]*4 + [3]*6 , None), # too many blacks at start
        ([BLACK]*3 + [WHITE]*5 + [BLACK] * 2 , None), # no space for 1
        ([WHITE] * 4 + [BLACK] *3 + [3] * 3, None), # no space for 2 or 1
        ([WHITE, BLACK, 3, 3, 3, 3, BLACK, BLACK, 3, BLACK],
         [WHITE, BLACK, BLACK, BLACK, WHITE, WHITE, BLACK, BLACK, WHITE, BLACK]),
        ([3, 3, BLACK, 3, 3, 3, BLACK, BLACK, 3, BLACK], 
         [BLACK, BLACK, BLACK, WHITE, WHITE, WHITE, BLACK, BLACK, WHITE, BLACK]),
        ([3, 3, 3, 3, BLACK, 3, BLACK, BLACK, 3, BLACK], 
        [WHITE, WHITE, BLACK, BLACK, BLACK, WHITE, BLACK, BLACK, WHITE, BLACK]), # right-most shift

    ]

    for arr, result in tests:
        m = matcher(arr, runs)
        print(m.match==result, m.match)


    run = (1, 1, 5)
    s = "---#--         -      # " # broken segment
    sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
    reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}
    row = [sym_map[x] for x in s]
    m = matcher(row, run)
    print("     ", s)
    m = ''.join([reverse_map[x] for x in m.match])
    print(m == "---#--#-----------#####-", m)
    m = matcher(row[::-1], run[::-1]).match[::-1]
    m = ''.join([reverse_map[x] for x in m])
    print(m == "---#-------------#-#####", m)
