"""
Lior Sinai, 17 October 2020

Unit tests for the matcher

"""

import unittest

BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

from matcher import find_match_backwards, find_match_forwards
from matcher_nfa import NonDeterministicFiniteAutomation


class TestMatcher(unittest.TestCase):
    def setUp(self):
        #self.matcher = find_match_backwards
        #self.matcher = find_match_forwards
        self.matcher = NonDeterministicFiniteAutomation().find_match

    def array_10(self):
        runs = (3, 2, 1)
        tests = [
            ([3] * 10 , [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 + [WHITE] *  2),
            ([BLACK]*4 + [3]*6 , None), # too many blacks at start
            ([BLACK]*3 + [WHITE]*5 + [BLACK] * 2 , None), # no space for 1
            ([WHITE] * 4 + [BLACK] *3 + [3] * 3, None), # no space for 2 or 1
            ([3, 3, BLACK, 3, 3, 3, BLACK, BLACK, 3, BLACK], 
            [BLACK, BLACK, BLACK, WHITE, WHITE, WHITE, BLACK, BLACK, WHITE, BLACK]),
            ([3, 3, 3, 3, BLACK, 3, BLACK, BLACK, 3, BLACK], 
            [WHITE, WHITE, BLACK, BLACK, BLACK, WHITE, BLACK, BLACK, WHITE, BLACK]), # right-most shift
        ]

        for arr, result in tests:
            arr = tuple(arr)
            m = self.matcher(arr, runs)
            self.assertEqual(m.match, result)
            
    def array_10_rightmost(self):
        runs = (3, 2, 1)
        row = (3, ) * 10
        m = self.matcher(row[::-1], runs[::-1])
        right_most =  [WHITE] *  2 + [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 
        self.assertEqual(m.match[::-1], right_most)

        row = (3, 3, 3, 3, 3, 3, 3, BLACK, 3, 3)
        right_most = [WHITE] *  2 + [BLACK]*3 + [WHITE] + [BLACK] * 2 + [WHITE] + [BLACK]*1 
        self.assertEqual(m.match[::-1], right_most)

    def long_run(self):
            runs = (10,)
            row = (WHITE,) + (EITHER,) + (BLACK, ) * 9 + (EITHER,) + (WHITE,) * 2
            left_most = [WHITE] +  [BLACK] * 10 +  [WHITE] * 3
            m = self.matcher(row, runs)
            self.assertEqual(m.match, left_most)

            m = self.matcher(row[::-1], runs[::-1])
            right_most = [WHITE] * 2 +  [BLACK] * 10 +  [WHITE] * 2
            self.assertEqual(m.match, right_most)

    def lancaster_example(self):
        #http://scc-forge.lancaster.ac.uk/open/nonogram/ls-fast
        run = (1, 1, 5)
        s = "---#--         -      # " # broken segment

        sym_map = {"-": WHITE, "#": BLACK, " ": EITHER}
        reverse_map = {WHITE:"-", BLACK:"#", EITHER:"?"}

        row = tuple(sym_map[x] for x in s)

        m = self.matcher(row, run).match
        m = ''.join([reverse_map[x] for x in m])
        self.assertEqual(m, "---#--#-----------#####-")

        m = self.matcher(row[::-1], run[::-1]).match[::-1]
        m = ''.join([reverse_map[x] for x in m])
        self.assertEqual(m, "---#-------------#-#####")
        
    def worst_case(self):
        pattern = (2, 5, 3, 2) 
        row = (EITHER,) * 19 + (BLACK,) # lots of backtracking because last result is false
        answer = [] 
        for p in pattern[:-1]:
            answer += [BLACK] * p + [WHITE]
        answer += [WHITE] * (len(row) - len(answer) - pattern[-1]) + [BLACK] * pattern[-1]

        m = self.matcher(row, pattern).match
        self.assertEqual(m, answer)
    
    def very_long(self):
        pattern = (2, 4, 9, 7, 1, 1, 8, 8, 3)
        row = (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
        ans = [1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        m = self.matcher(row, pattern).match
        self.assertEqual(m, ans)


def suite():
    "Set order of tests in ascending order of complexity and code required"
    suite = unittest.TestSuite()
    # functions test
    suite.addTest(TestMatcher('array_10'))
    suite.addTest(TestMatcher('array_10_rightmost'))
    suite.addTest(TestMatcher('long_run'))
    suite.addTest(TestMatcher('lancaster_example'))
    suite.addTest(TestMatcher('worst_case'))
    suite.addTest(TestMatcher('very_long'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
