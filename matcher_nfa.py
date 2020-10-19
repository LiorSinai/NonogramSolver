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

from Match import Match, minimum_sequence

class State():
    def __init__(self, value, prev=None, id=0, qualifier=None):
        self.prev = prev
        self.next  = None
        self.next1 = None # only used if a split
        self.qualifier = qualifier
        
        self.value = value
        self.is_final = False # only true in the end state

        self.id = id

    def transition(self, value):
        if self.qualifier:
            # make a split
            s1 = self.next if (self.next.value & value) else None
            if self.next1:
                s2 = self.next1 if (self.next1.value & value) else None
            else:
                s2 = None
            return s1, s2         
        elif self.next.value == self.next.value & value:
            return self.next, self.next1
        else:
            return None




class NonDeterministicFiniteAutomation():
    def __init__(self):
        self.states = []
        self.state_id = 0 # used for efficient movement already the list. Because of splits, this is non-linear
        self.pattern = ()

    def convert_pattern(self, pattern):
        fragments = [WHITE, "*"]
        for p in pattern:
            fragments += [BLACK] * p + [WHITE, "+"]
        fragments = fragments[:-2] # skip the last white *
        return fragments

    def construct(self, pattern):
        # reset self
        self.pattern = pattern
        self.states = [] 
        self.state_id = 0
        
        start_state = State(None, prev=None, id=self.state_id, qualifier ="start")
        self.states.append(start_state)

        for symbol in self.convert_pattern(pattern):
            self.add_state(symbol)
        self.states[-1].is_final = True  

        for state in self.states:
            if state.qualifier == "*":  
                prev_state = state.prev
                next_state = state.next
                prev_state.next  = next_state # option to skip this state
                prev_state.next1 = state      # this was prev_state.next originally. Moved to find minimum match (blacks match first)

    def add_state(self, symbol):
        prev_state = self.states[-1]
        if symbol in ["+", "*"]:
            prev_state.next1 = prev_state  # loop back to self
            prev_state.qualifier = symbol
        else:
            self.state_id += 1  
            state = State(symbol, prev=prev_state, id=self.state_id)
            self.states.append(state)
            prev_state.next = state  # patch previous state

    def find_minimum_match(self, array, pattern):
        """ breadth first search, very slow, O(2^n) time """

        # special case optimisation
        if not pattern:
            # match an empty sequence
            if array.count(BLACK) == 0:
                match = [WHITE] * len(array)
                return Match(match, pattern=self.pattern)     
            else:
                return Match(pattern=self.pattern) #no match
        min_length = sum(pattern) + (len(pattern) - 1)
        if array.count(BLACK) == 0 and array.count(WHITE) ==0:
            # construct minimum pattern
            return Match(minimum_sequence(pattern, len(array)), pattern=self.pattern)


        self.construct(pattern) # create the state firsts

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
                if next_state is not None:
                    s1, s2 = next_state # possible split
                    for s in next_state:
                        if s and s.is_final:
                            if array[idx+1:].count(BLACK) == 0:
                                match_final = match + [s.value]
                                match_final += [WHITE] * (len(array) - idx - 1)
                                return Match(match_final, pattern=self.pattern)
                            # else its not added to the stack
                        elif s:
                            stack.append(s.id)
                            matches.append(match + [s.value])

        return Match(pattern=self.pattern) # no match


    def find_match(self, array, pattern):
        """ finds a minimum length match, not necessarily the left-most. Very fast, O(n^2) time """

        # special case optimisation
        if not pattern:
            # match an empty sequence
            if array.count(BLACK) == 0:
                match = [WHITE] * len(array)
                return Match(match, pattern=self.pattern)     
            else:
                return Match(pattern=self.pattern) #no match
        min_length = sum(pattern) + (len(pattern) - 1)
        if array.count(BLACK) == 0 and array.count(WHITE) ==0 or not pattern:
            # construct minimum pattern
            return Match(minimum_sequence(pattern, len(array)), pattern=self.pattern)

        self.construct(pattern) # create the state firsts

        # simulate finite state machine. Only keeps one path, not necessarily the best
        idx = - 1
        stack = {0: []} # key  
        while idx < len(array) - 1 and stack:
            idx += 1
            state_ids = list(stack.keys())
            for state_id in state_ids:
                # advance each one at a time
                match = stack.pop(state_id)
                state = self.states[state_id]
                next_state = state.transition(array[idx])
                if next_state is not None:
                    s1, s2 = next_state # possible split
                    for s in next_state:
                        if s and s.is_final:
                            if array[idx+1:].count(BLACK) == 0:
                                match_final = match + [s.value]
                                match_final += [WHITE] * (len(array) - idx - 1)
                                return Match(match_final, pattern=self.pattern)
                            # else its not added to the stack
                        elif s and s.id not in stack:
                            stack[s.id] = match + [s.value]

        return Match(pattern=self.pattern) # no match

                

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
        m = matcher(arr, runs)
        print(m.match==result, m.match)


    run = (1, 1, 5)
    s = "---#--         -      # " # broken segment
    sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
    reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}
    row = [sym_map[x] for x in s]
    m = matcher(row, run)
    m = ''.join([reverse_map[x] for x in m.match])
    print(m == "---#--#-----------#####-", m)
    m = matcher(row[::-1], run[::-1]).match[::-1]
    m = ''.join([reverse_map[x] for x in m])
    print(m == "---#-------------#-#####", m)
