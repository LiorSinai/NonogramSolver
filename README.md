# Nonogram Solver - C++

A Nonogram solver in Python. See the Wikipeida [article][nonogram_wiki].

[nonogram_wiki]: https://en.wikipedia.org/wiki/Nonogram 

Various components:
- a slow complete solver.
- a fast heuristic solver. May require guessing.
- output to the screen, file, or a Matplotlib image plot.


## Algorithm 

### Complete solver

Based on ideas by [Rosetta Code][rosetta_code].

Algorithm
- Generate all possible rows and columns. This is done using unique integer partition permutations. This is very slow and memory intenstive.
- Do constraint propagation with row and column sweeps.
	- Discard possible rows/columns which do not fit in the current puzzle.
	- Determine if there are any cells that are the same colour in all possibilities. 
- If there are more than 1 possible arrangements for any row/column after constraint propagation, try place possible rows, and check if some possible column fits. This can be very slow and memory intenstive. 

### Fast solver
Based on ideas by [Jan Wolter][Wolter_survey] and [Steve Simpson][lancaster_solver].

Algorithm
- Do constraint propagation with row and column sweeps.
	- Find the left-most and right-most match for each row/column. This is done using a Nondeterministic Finite State Machine and Thompsons algorithm. This is an O(n^2) algorithm. See my blog [post][nfa_post] for more detail.
	- Find overlaps. This is done by finding a "changer sequence". See the example below. It is made by replacing each symbol with a counter which increments everytime the symbols change e.g. from BLACK to WHITE. Counter values which are in the same index in both left and right matches are overlaps.
	- A simple filler which adds very simple clues that the left-right algorithm sometimes misses e.g. a white after ending a black sequence.
- If this fails, guess and then go back to constraint propagation.
	- First try to find contradictions -> guesses which are obviously wrong. This will happen if there are no matches for the line matcher. The opposite guess is therefore correct.
	- Otherwise save the current grid and make a binary guess. Backtrack if it is wrong. This is O(2^n), so it can very easily lead down a never ending spiral of guesses. 
current grid and make a binary guess. Backtrack if it is wrong. This is O(2^n), so it can very easily lead down a never ending spiral of guesses. 

<pre>
Example by [Steve Simpson][lancaster_fast]:	
Length:   24 
Rule:     1,1,5 
Original: >---#---------#-----#####< 
Broken:   >---#--         -      # < 
      left>000122344444444444555556< 
     right>000122222222222223455555< 
      fast>---#--+++++++++-+++####+< 
</pre>

[Wolter_survey]: https://webpbn.com/survey/
[lancaster_solver]: http://scc-forge.lancaster.ac.uk/open/nonogram/
[lancaster_fast]: http://scc-forge.lancaster.ac.uk/open/nonogram/ls-fast
[rosetta_code]: https://rosettacode.org/wiki/Nonogram_solver
[nfa_post]: https://liorsinai.github.io/coding/2020/10/29/finite-state-machines.html

## Input files

### Solve collection
Solve a collection of puzzles in a single file.

Format is:

---
<div>
General description (this line is skipped by the file reader) <br>
-- puzzle title (this line is skipped by the file reader)<br>
[run rows] <br>
[run cols] <br>
-- next puzzle title (this line is skipped by the file reader) <br>
... <br>
--- 
</div>

The runs are given by sets of unicode characters, with A=1, B=2 and so on. Spaces represent a new row/column.
This can work for a numbers greater than 26, but these will be encoded as non-Basic Latin unicode symbols.

### Solve puzzle
Based on Steve Simpsons .non format.

Format is:

---
<div>
text    (this line is skipped by the file reader) <br>
title  (this line is skipped by the file reader) <br>

width w (this line is skipped by the file reader) <br>
height h (this line is skipped by the file reader) <br>

rows  <br>
x, x, x <br>
.... <br>

columns <br>
x, x, x <br>
... <br>
</div>

---

## Example solutions
Elephant, 15x15, solved in 0.012s
<pre>
-  -  -  -  -  -  03 -  -  -  01 -  -  -  -  -  -  -  -   
-  -  -  -  -  -  03 07 -  -  05 02 -  -  01 01 01 01 -  
-  -  -  -  01 11 01 02 07 15 07 08 14 09 06 09 09 10 12 
-  -  -  03 .  .  .  .  .  #  #  #  .  .  .  .  .  .  . 
-  -  04 02 .  .  #  #  #  #  .  #  #  .  .  .  .  .  .   
-  -  06 06 .  #  #  #  #  #  #  .  #  #  #  #  #  #  . 
-  06 02 01 .  #  #  #  #  #  #  .  #  #  .  .  .  .  # 
01 04 02 01 .  #  .  #  #  #  #  .  #  #  .  .  .  .  # 
-  06 03 02 .  #  #  #  #  #  #  .  #  #  #  .  .  #  # 
-  -  06 07 .  #  #  #  #  #  #  .  #  #  #  #  #  #  # 
-  -  06 08 #  #  #  #  #  #  .  #  #  #  #  #  #  #  # 
-  -  01 10 .  #  .  .  .  #  #  #  #  #  #  #  #  #  # 
-  -  01 10 .  #  .  .  .  #  #  #  #  #  #  #  #  #  # 
-  -  01 10 .  #  .  .  .  #  #  #  #  #  #  #  #  #  # 
01 01 04 04 .  #  .  #  .  #  #  #  #  .  .  #  #  #  # 
-  03 04 04 .  #  #  #  .  #  #  #  #  .  .  #  #  #  # 
-  -  04 04 .  .  .  .  .  #  #  #  #  .  .  #  #  #  #   
-  -  04 04 .  .  .  .  .  #  #  #  #  .  .  #  #  #  # 
</pre>

"Where there is smoke", solved in 0.154s with 12 guesses:
<div>
<pre>
- -  -  -  -  -  -  -  -  -  -  01 -  -  -  -  -  -  -  -  -  -  -  -  -  - 
- -  -  -  -  -  -  -  03 -  -  01 02 -  03 -  -  -  -  -  -  -  -  -  -  - 
- -  -  -  -  -  -  01 03 -  01 01 01 02 01 01 -  -  -  -  -  03 -  -  -  - 
- -  -  -  -  -  02 06 01 -  03 02 01 04 02 04 03 02 -  -  -  02 03 -  -  01 
- -  -  -  -  -  02 04 01 02 03 01 01 03 03 02 01 01 03 07 05 01 04 09 08 08 
- -  -  -  -  -  01 04 04 02 03 02 01 03 01 01 02 01 03 04 04 03 01 02 03 02 
- -  01 03 02 01 .  #  .  #  #  #  .  .  #  #  .  .  .  .  .  .  .  .  .  #
- -  -  01 02 02 #  .  #  #  .  .  .  #  #  .  .  .  .  .  .  .  .  .  .  .
- -  -  -  03 04 #  #  #  .  .  .  #  #  #  #  .  .  .  .  .  .  .  .  .  .
- -  -  02 03 02 .  #  #  .  #  #  #  .  .  #  #  .  .  .  .  .  .  .  .  .
- -  -  02 01 06 #  #  .  .  #  .  .  .  .  #  #  #  #  #  #  .  .  .  .  .
- -  -  02 13 01 #  #  .  .  #  #  #  #  #  #  #  #  #  #  #  #  #  .  .  #
- -  -  01 01 08 .  #  .  .  .  .  .  #  .  .  .  .  #  #  #  #  #  #  #  #
- -  02 01 01 07 .  #  #  .  .  .  .  #  .  #  .  .  .  #  #  #  #  #  #  #
- 01 02 02 02 03 .  .  #  .  .  .  #  #  .  #  #  .  .  #  #  .  .  #  #  #
- 03 01 01 01 03 #  #  #  .  .  #  .  .  .  .  .  .  .  #  .  #  .  #  #  #
- 01 02 01 01 03 .  #  .  .  #  #  .  .  .  .  .  .  .  #  .  #  .  #  #  #
- -  02 01 01 03 .  #  #  .  #  .  .  .  #  .  .  .  .  .  .  .  .  #  #  #
- -  -  01 05 05 .  #  .  .  #  #  #  #  #  .  .  .  .  .  .  #  #  #  #  #
- -  -  01 01 03 .  .  #  .  .  .  .  #  .  .  .  .  .  .  .  .  #  #  #  .
- -  -  -  04 02 .  .  .  .  .  #  #  #  #  .  .  .  .  .  .  .  #  #  .  .
- 02 02 01 02 01 .  #  #  .  #  #  .  .  #  .  .  .  .  .  .  #  #  .  .  #
- 02 01 02 03 02 .  #  #  .  #  .  .  #  #  .  .  .  .  #  #  #  .  .  #  #
- -  04 01 06 01 .  #  #  #  #  .  .  #  .  .  #  #  #  #  #  #  .  .  #  .
- -  03 04 03 02 .  #  #  #  .  .  .  #  #  #  #  .  #  #  #  .  .  #  #  .
- -  -  -  04 02 .  .  .  .  .  .  .  .  .  .  .  #  #  #  #  .  #  #  .  .
</pre>
</div>


Bear, 40x50, solved in 0.8s:
<div>
<pre>
# # # # # # # # . . . . # # # # # # # # # # # # # # # # # # # # # # # # # # # # # . . . . . # # # #
# # # # # # . . # # # # . . # # # # # # # # # # # # # # # # # # # # # # # # # . . # # # # . . # # #
# # # # # . # # # . . # # . . # # # . . . . # # # # # # # # # . . . # # # # . # # . . . . # . # # #
# # # # . # # . . . . . # # . . . . # # . . . . . . . . . . . # # . . . . . # . . . . . . # # . # #
# # # # . # . . . . . . . . # . # # # # # # # # # . . . # # # # # # # # # # . # # . . . . . # # . #
# # # . # # . . . . . # # # # # # . . . . . . . # # # # # . . . . . . . # # # # # . . . . . . # . #
# # # . # . . . . . # # # # # . . . . . . . . . . . . . . . . . . . . . . . # # # # # . . . . # . #
# # # . # . . . . # # # # . . . . . . . . . . . . . . . . . . . . . . . . . . # # # # . . . . # . #
# # # . # . . . # # # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # # # . . . # . #
# # # . # . . . # # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # # . . # . . #
# # # . # . . # # # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . # # # # # # . # #
# # # . . # . # # . # # # . . . . . . . . . . . . . . . . . . . . . . . . . . # # . # # # # . . # #
# # # # . # # # . . # . # # # # # # # # . . . . . . . . . . . . . # # # # # # # . # . # # . . # # #
# # # # . . # # . . # . # # # # # # # # # # # # . . . . . # # # # # # # # # # # . # . # # . # # # #
# # # # # . # . # # . . # # # # # # # . . # # . . . . . . . # # . . # # # # # # . . # . # . # # # #
# # # # . . # . # . . # . . # # # # # # . # # . . . . . . . # # . # # # # # # . # . # # . # . # # #
# # # # . # . # . . # # . . # # # # . # # # . . . . . . . . # # # # . # # # . # . # . # . # . # # #
# # # # . # . # . # # . . # . # # # # . . # . . . . . . . . # # . . # # # . # # . # . . # # . . # #
# # # . . # . # . # . . # # . . . # # # # # . . . . . . . . # # # # # # . # . # . . # . # # # . # #
# # # . # # . # . # . # # . . # . # # # # # . . # # # # . . . # # # # . # # . . # . . # # . # . # #
# # # . # # . # # . . # . . # . . . # # # # . # # . . . # # . # # # . # . . # . # # . # . . # . # #
# # # . # . # # # . # # . . # . # . # # # # . # . # # # # # . # # # . # # . . # . . # # # . # . # #
# # # . # . . # # . # . . # # . # . . # # # . # # # # # # # . # # # # . . # . . # # # # . # # . # #
# # . . # . . # # # # . # . . # . # . # # . . # # # # # # . . # # . # # . # # . . # # # . # # . . #
# # . # # . # # # # . . # . # # . # . # # . . . # # # # # . . # # . . . # . . # . # # # . . # # . #
# # . # # . # . # # # # . . # . # . # # # . . . . # # # . . . . # # . . # . # # # # . . # # # # . #
# # . # # # . . . # # # . # # . # . # # # . . . . . . . . . . . # # # . # # # # # # # . # # # # . #
# # . # # # . # # . # # # # . . # # # # # . . . . . . . . . . . # # # # # # # # . . # . # # . # . .
# . . # . # # # . . # # # # # # # # # # # . . # # # # # # . . . # # # # # # # . # . . # # # . # . .
# . . # . # # . . # # . # # # # # # # # # # # # # . . . # # # # # # # # # # . . # # . # # # . # # .
# . # # . . # # # . . . # . # # # # # # . # . # . . . . . # # # # # # # . . # . . # # # # # . # # .
# . # . . . # # # . # # . . # # # # # # . # . . # . . . # . # . # # # # . . . # . # # # # . . # # .
# . # . . # # # # # # . . # # # # # # # . # # . # # # # . # # . # # # # # . . # # # # # # . . . # .
# . # . # # . . # # # . . # . . # # # # . . # # . . . . # # . . # # # # # # # # # # # . # # . . # .
# . # . # . . # . # # . # . . . # # # # # . . . . . . . . . . # # # # # # # # # # . # . . # . . # .
# . # . # . . # . # # # # . . # # # # # # # . . # # # # . . # # # # # # # # # # . . # . . # . . # .
# . # # . . # . . # . # # # # # # # # # # # # # # # # # # # # # # # # # # # # . # . . # . . # # # .
# . # # . . # . # # . # # # # # # # # # # # # # # # # # # # # # # # # # # # . . # # . # . . # # # .
# . . # . # . . # . . . . # # # # # # # # # # # # # # # # # # # # # # # # # # . . # . . # . . # . #
# # . # # # . . # . . . # # # # # # # # # # # # # # # # # # # # # # # # # # # # . # # . # . # # . #
</pre>
</div>
