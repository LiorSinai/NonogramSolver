"""
Lior Sinai, 19 October 2020

Non-deterministic finite automation matcher using Thompson's algorithm

See
-- Regular Expression Matching Can Be Simple And Fast at https://swtch.com/~rsc/regexp/regexp1.html
-- https://kean.blog/post/regex-compiler#one-or-more-quantifier and https://kean.blog/post/regex-matcher 


Uses breadth-first search instead of depth-first search.
But still has the same limitations ...

If don't need to construct the match, can use states and spend up time hugely

"""

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

max_size_cache = 5000

from match import Match, minimum_sequence, special_matches
from functools import lru_cache

class State():
    def __init__(self, symbol, id=0, qualifier=None):
        self.symbol = symbol
        self.id = id
        self.qualifier = qualifier # purely a description, not for functional purposes
        
        self.transitions = []
        self.is_final = False # only true in the end state

    def transition(self, symbol):
        next_states = []
        for state in self.transitions:
            if state.symbol & symbol:
                next_states.append(state)
        return next_states


class NonDeterministicFiniteAutomation():
    def __init__(self):
        self.states = []
        self.state_id = 0 # used for efficient movement already the list. Because of splits, this is non-linear
        self.pattern = ()

    def convert_pattern(self, pattern):
        fragments = [WHITE, "*"]
        for p in pattern[:-1]:
            fragments += [BLACK, "."] * p + [WHITE, "+", "."]
        if pattern:
            fragments += [BLACK, '.'] * pattern[-1] # skip the last white *
        return fragments

    def compile(self, pattern):
        # reset self
        self.pattern = pattern
        self.states = [] 
        
        start = State(None,  id=self.state_id, qualifier ="start")
        end   = State(None, id=1) # continiously update end, since Python doesn't allow dangling pointers
        self.states = [start, end]

        self.state_id = 1
        stack = [end, start] 
        for symbol in self.convert_pattern(pattern):
            self.add_state(symbol, stack)
        self.states  = self.states[:-1] # chop off the last unneccessary end
        self.states[-1].is_final = True  

    def add_state(self, symbol, stack):
        if symbol in ["+", "*"]:  # one or more
            state = stack[1]
            state.transitions.append(state) # loop back to self
            state.qualifier = symbol
            if symbol == "*": # zero or more
                state = stack.pop(1)  # remove this state because it'll be skipped
                prev_state = stack[1]
                next_state = stack[0]
                state.transitions.insert(0, next_state)
                prev_state.transitions.insert(0, state)
        elif symbol == ".": # concatenation
            state = stack[1]
            prev_state = stack.pop(2)
            prev_state.transitions.insert(0, state)
        else:
            # change end to a state
            state = stack[0]
            state.symbol = symbol
            # add a new end
            self.state_id += 1  
            end = State(None, id=self.state_id)
            self.states.append(end)
            stack.insert(0, end)

    def find_match_BFS(self, array, pattern):
        """ breadth first search, very slow, O(2^n) time """

        # special case optimisation
        match = special_matches(array, pattern)
        if match.is_match:
            return match

        self.compile(pattern) # create the state firsts

        # simulate finite state machine
        idx = - 1
        stack = [0]   # a stack of current states
        matches = [[]] # a stack of corresponding matches
        while idx < len(array) - 1 and stack:
            idx += 1
            K = len(stack) # note this this grows with splits
            for k in range(K):
                # advance each one at a time
                state_id = stack.pop(0)
                match = matches.pop(0)
                state = self.states[state_id]
                next_state = state.transition(array[idx])
                for s in next_state:
                    if s and s.is_final:
                        if array[idx+1:].count(BLACK) == 0:
                            match_final = match + [s.symbol]
                            match_final += [WHITE] * (len(array) - idx - 1)
                            return Match(match_final, pattern=self.pattern)
                        # else: its not added to the stack
                    elif s:
                        stack.append(s.id)
                        matches.append(match + [s.symbol])

        return Match(pattern=self.pattern) # no match

    @lru_cache(maxsize=max_size_cache) # Least Recently Used Cache
    def find_match(self, array, pattern):
        """ finds a minimum length, left-most match. Very fast, O(n*m) time """
        self.compile(pattern) # create the states first
        min_length = len(self.states) - 2 #= sum(pattern) + len(pattern) -1 

        # simulate finite state machine. Only keeps one path per state.
        idx = - 1
        stack = {0: []} # state_id: match  
        new_stack = {}
        while idx < len(array) - 1 and stack:
            idx += 1
            for state_id, match in stack.items():
                # advance each one at a time
                state = self.states[state_id]
                for s in state.transitions:
                    if s.symbol & array[idx]:
                        if s.is_final:
                            if array[idx+1:].count(BLACK) == 0:
                                match_final = match + [s.symbol]
                                match_final += [WHITE] * (len(array) - idx - 1)
                                return Match(match_final, pattern=self.pattern)
                            # else: its not added to the stack
                        elif (s.id==state.id or s.id not in new_stack):# and (len(array) - (idx)) >= (min_length - s.id + 1):
                            new_stack[s.id] = match + [s.symbol]
            stack = new_stack;
            new_stack = {};

        return Match(pattern=self.pattern) # no match

    def find_match_DFS(self, array, pattern):
        """ Depth first search, very slow"""
        def simulate(state_id, match, idx):
            if idx >= len(array):
                return Match(pattern=self.pattern) # no match
            state = self.states[state_id]
            for s in state.transitions:
                if s.symbol & array[idx]:
                    if s.is_final:
                        if array[idx+1:].count(BLACK) == 0:
                            match_final = match + [s.symbol]
                            match_final += [WHITE] * (len(array) - idx - 1)
                            return Match(match_final, pattern=self.pattern)
                        # else: its not added to the stack
                    else:
                        ans = simulate(s.id, match + [s.symbol], idx+1)
                        if ans.is_match:
                            return ans
            return Match(pattern=self.pattern) # no match
        min_length = sum(pattern) + len(pattern) -1
        self.compile(pattern) # create the state first

        return simulate(0, [], 0) # start recursive call

def find_match(array, pattern):
    # special case optimisation
    match = special_matches(array, pattern)
    if match.is_match:
        return match
    nfa = NonDeterministicFiniteAutomation()
    return nfa.find_match(array, pattern)
            
            
if __name__ == '__main__':
    matcher = NonDeterministicFiniteAutomation().find_match

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
        m = matcher(tuple(arr), runs)
        print(m.match==result, m.match)


    run = (1, 1, 5)
    s = "---#--         -      # " # broken segment
    sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
    reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}
    row = tuple([sym_map[x] for x in s])
    m = matcher(row, run)
    m = ''.join([reverse_map[x] for x in m.match])
    print(m == "---#--#-----------#####-", m)
    m = matcher(row[::-1], run[::-1]).match[::-1]
    m = ''.join([reverse_map[x] for x in m])
    print(m == "---#-------------#-#####", m)

    # import re
    # pattern = "[- ]*"
    # for p in run:
    #     pattern += "([# ]){" + str(p) + "}" + "([- ]+)"
    # pattern = pattern[:-2] + "*)"
    # m = re.fullmatch(pattern, s)
    # print(m)