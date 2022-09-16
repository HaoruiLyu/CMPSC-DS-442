############################################################
# CMPSC 442: Homework 2
############################################################

student_name = "Haorui Lyu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

from math import factorial
from queue import Queue
import random
import copy
'''
from itertools import permutations
'''

############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    # Use C(n*n, n) to solve this question
    # The formula is (n * n)! / ((n * n - n)! * n!)
    return int(factorial(n * n) / (factorial(n * n - n) * factorial(n)))

def num_placements_one_per_row(n):
    return n ** n

def n_queens_valid(board):
    '''
    >>> n_queens_valid([0, 0]) 
    False 
    >>> n_queens_valid([0, 2]) 
    True 
    >>> n_queens_valid([0, 1]) 
    False 
    >>> n_queens_valid([0, 3, 1]) 
    True 
    '''
    column_list = []
    for i in range(len(board)):
        if board[i] not in column_list: # Check if some queens in same column
            for j in range(len(column_list)):
                # Check if queen in same diagonal
                # if row_1 - row_2 = col_1 - col_2, their are in same diagonal
                if board[i] == column_list[j] - (i - j) or board[i] == column_list[j] - (j - i):
                        return False
            column_list.append(board[i])
        else:
            return False
    return True        
    
'''

def n_queens_solutions(n):
    possible = map(list, list(permutations(range(n))))
    for board in possible:
        if n_queens_valid(board):
            yield board
            
# Another solution, but the time complexity is relatively high            

'''
            
def n_queens_solutions(n):
    '''
    >>> solutions = n_queens_solutions(4) 
    >>> next(solutions) 
    [1, 3, 0, 2] 
    >>> next(solutions) 
    [2, 0, 3, 1] 
    >>> list(n_queens_solutions(6)) 
    [[1, 3, 5, 0, 2, 4], [2, 5, 1, 4, 0, 3], 
    [3, 0, 4, 1, 5, 2], [4, 2, 0, 5, 3, 1]] 
    >>> len(list(n_queens_solutions(8))) 
    92 
    '''
    res = []
    def n_queens_helper(i = 0, col = [], l_diag = set(), r_diag = set()):
        if i == n:
            res.append(col)
            return
        for j in range(n):
            if j not in col and i - j not in l_diag and i + j not in r_diag:
                n_queens_helper(i+1, col+[j], l_diag|{i-j}, r_diag|{i+j} )
    n_queens_helper()
    for i in res:
        yield i
            
############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])

    def get_board(self):
        '''
        >>> b = [[True, False], [False, True]]
        >>> p = LightsOutPuzzle(b)
        >>> p.get_board()
        [[True, False], [False, True]]
        >>> b = [[True, True], [True, True]]
        >>> p = LightsOutPuzzle(b)
        >>> p.get_board()
        [[True, True], [True, True]]
        '''
        return self.board

    def perform_move(self, row, col):
        '''
        >>> p = create_puzzle(3, 3)
        >>> p.perform_move(1, 1)
        >>> p.get_board()
        [[False, True, False], [True, True, True], [False, True, False]]
        >>> p = create_puzzle(3, 3)
        >>> p.perform_move(0, 0)
        >>> p.get_board()
        [[True, True, False], [True, False, False], [False, False, False]]
        '''
        dira = [[0,0],[-1,0],[1,0],[0,1],[0,-1]]
        for i in dira:
            x_move = row + i[0]
            y_move = col + i[1]
            if 0 <= x_move < self.row and 0 <= y_move < self.col:
                self.board[x_move][y_move] ^= True

    def scramble(self):
        for i in range(self.row):
            for j in range(self.col):
                if random.random() < 0.5:
                    self.perform_move(i, j)

    def is_solved(self):
        '''
        >>> b = [[True, False], [False, True]]
        >>> p = LightsOutPuzzle(b)
        >>> p.is_solved()
        False
        >>> b = [[False, False], [False, False]]
        >>> p = LightsOutPuzzle(b)
        >>> p.is_solved()
        True
        '''
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j]:
                    return False
        return True
                    

    def copy(self):
        '''
        >>> p = create_puzzle(3, 3)
        >>> p2 = p.copy()
        >>> p.get_board() == p2.get_board()
        True
        >>> p = create_puzzle(3, 3)
        >>> p2 = p.copy()
        >>> p.perform_move(1, 1)
        >>> p.get_board() == p2.get_board()
        False
        '''
        return copy.deepcopy(self)

    def successors(self):
        '''
        >>> p = create_puzzle(2, 2)
        >>> for move, new_p in p.successors():
        ...     print(move, new_p.get_board())
        (0, 0) [[True, True], [True, False]]
        (0, 1) [[True, True], [False, True]]
        (1, 0) [[True, False], [True, True]]
        (1, 1) [[False, True], [True, True]]
        >>> for i in range(2, 6):
        ...     p = create_puzzle(i, i + 1)
        ...     print(len(list(p.successors())))
        6
        12
        20
        30
        '''
        for i in range(self.row):
            for j in range(self.col):
                cop = self.copy()
                cop.perform_move(i, j)
                yield ((i,j),cop)

    def find_solution(self):
        '''
        >>> p = create_puzzle(2, 3)
        >>> for row in range(2):
        ...     for col in range(3):
        ...         p.perform_move(row, col)
        >>> p.find_solution()
        [(0, 0), (0, 2)]
        >>> b = [[False, False, False],
        ... [False, False, False]]
        >>> b[0][0] = True
        >>> p = LightsOutPuzzle(b)
        >>> p.find_solution() is None
        True
        '''
        vit = set() # record the encountered state
        q = Queue() # import FIFP queue
        q.put([self,[]]) # put Queue into the board
        while not q.empty():
            info = q.get()
            cur, moves = info[0], info[1]
            # get the board and move from top of the Queue
            if cur.is_solved():
                return moves
            strs = "" # transfor state to str and record
            for i in range(cur.row):
                for j in range(cur.col):
                    strs += str(cur.board[i][j])
            if strs in vit: # if state has been visited skip it
                continue
            vit.add(strs) # if not be visited, record it
            for move, x_move in cur.successors():
                # visit the next state for board right now and record it
                q.put([x_move, moves + [move]])
        return None
        

def create_puzzle(rows, cols):
    '''
    >>> p = create_puzzle(2, 2)
    >>> p.get_board()
    [[False, False], [False, False]]
    >>> p = create_puzzle(2, 3)
    >>> p.get_board()
    [[False, False, False], [False, False, False]]
    '''
    return LightsOutPuzzle([[False] * cols for i in range(rows)])

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):
    '''
    >>> solve_identical_disks(4, 2)
    [(0, 2), (1, 3)]
    >>> solve_identical_disks(5, 2)
    [(0, 2), (1, 3), (2, 4)]
    >>> solve_identical_disks(4, 3)
    [(1, 3), (0, 1)]
    >>> solve_identical_disks(5, 3)
    [(1, 3), (0, 1), (2, 4), (1, 2)]
    '''
    vis = set() # record the encountered state
    pos = [1] * n + [0] * (length - n)
    # initialization state, first n disks are 1, followed by 0
    q = Queue()
    q.put([pos,[]])
    ans = None # record final answer
    while not q.empty():
        info = q.get()
        pos,moves = info[0],info[1] 
        # Get the state and the steps to get to that state
        if sum(pos[length - n:]) == n:
            # If all 1's have reached the last n positions, 
            # find the answer, return to this state step
            ans = moves
            break
        hs = "" # Turn this state into a string for storage
        for v in pos:hs += str(v)
        if hs in vis:continue # If the state already exists skip it 
        vis.add(hs) # Otherwise join the state
        for i in range(length):
            if pos[i] == 0:
                continue # skip empty positions
            if i+1<length and pos[i+1] == 0:
                # The current disk can be moved to the first position on the right
                nx = copy.deepcopy(pos)
                nx[i],nx[i+1] = 0,1 # exchange content
                q.put([nx,moves + [(i,i+1)]]) # Put state and steps
            if i+2<length and pos[i+1] == 1 and pos[i+2] == 0:
                # The current disk can be moved to the second position on the right
                nx = copy.deepcopy(pos)
                nx[i],nx[i+2] = 0,1 # exchange content
                q.put([nx,moves + [(i,i+2)]]) # Put state and steps
    return ans

def solve_distinct_disks(length, n):
    '''
    >>> solve_distinct_disks(4, 2)
    [(0, 2), (2, 3), (1, 2)]
    >>> solve_distinct_disks(5, 2)
    [(0, 2), (1, 3), (2, 4)]
    >>> solve_distinct_disks(4, 3)
    [(1, 3), (0, 1), (2, 0), (3, 2), (1, 3), (0, 1)]
    >>> solve_distinct_disks(5, 3)
    [(1, 3), (2, 1), (0, 2), (2, 4), (1, 2)]
    '''
    vis = set() # record the encountered state
    first_pos = [0] * length
    for i in range(length):
        # Initialization state, the first n disks are numbered, followed by 0
        if i < n:first_pos[i] = i + 1
    q = Queue()
    q.put([first_pos, []])
    ans = None # record final answer
    while not q.empty():
        info = q.get()
        pos, moves = info[0], info[1] 
        # Get the state and the steps to get to that state
        if first_pos == pos[::-1]:
            # If the state is the opposite of the initial state, 
            # find the answer and return to the steps to reach this state
            ans = moves
            break
        hs = "" # Turn this state into a string for storage
        for v in pos: hs += str(v)
        if hs in vis: continue # If the state already exists skip it 
        vis.add(hs) # Otherwise join the state
        for i in range(length):
            if pos[i] == 0: continue # skip empty positions
            if i + 1 < length and pos[i + 1] == 0:
                # The current disk can be moved to the first position on the right
                nx = copy.deepcopy(pos)
                nx[i], nx[i + 1] = nx[i + 1], nx[i]
                q.put([nx, moves + [(i, i + 1)]])
            if i + 2 < length and pos[i + 1] != 0 and pos[i + 2] == 0:
                # The current disk can be moved to the second position on the right
                nx = copy.deepcopy(pos)
                nx[i], nx[i + 2] = nx[i + 2], nx[i]
                q.put([nx, moves + [(i, i + 2)]])
            if i - 1 >= 0 and pos[i - 1] == 0:
                # The current disk can be moved to the first position on the left
                nx = copy.deepcopy(pos)
                nx[i], nx[i - 1] = nx[i - 1], nx[i]
                q.put([nx, moves + [(i, i - 1)]])
            if i - 2 >= 0 and pos[i - 1] != 0 and pos[i - 2] == 0:
                # The current disk can be moved to the second position on the left
                nx = copy.deepcopy(pos)
                nx[i], nx[i - 2] = nx[i - 2], nx[i]
                q.put([nx, moves + [(i, i - 2)]])
    return ans

############################################################
# Section 4: Feedback
############################################################

feedback_question_1 = """
I spent about 14 hours.
"""

feedback_question_2 = """
The find_solution(self) function of Part2 and the entire Part3. 
I haven't implemented BFS methods in Python before, I've scoured the web for 
a lot of tutorials on BFS. This is a big challenge for me, because the professor 
in the previous CMPSC465 class just gave us pseudo-code, and I need to implement
BFS by combining pseudo-code and online materials.
"""

feedback_question_3 = """
I really like the GUI part of Part2, 
I think it's cool to implement a small game-like program in code.

In addition, for the n_queens_solutions(n) function of Part1, 
what I thought at first was to use permutations to randomly generate 
all the positions and then verify whether these positions are feasible 
one by one. But I think the time and space complexity of this method 
is too high. Later I used backtracking to solve this problem. 
I think that's what I've learned as well, not just simply doing the homework, 
but thinking about how to optimize the code.
"""
