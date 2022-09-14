############################################################
# CMPSC 442: Homework 2
############################################################

student_name = "Haorui Lyu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

from math import factorial
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

'''
            
def n_queens_solutions(n):
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

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        pass

    def scramble(self):
        pass

    def is_solved(self):
        pass

    def copy(self):
        pass

    def successors(self):
        pass

    def find_solution(self):
        pass

def create_puzzle(rows, cols):
    pass

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):
    pass

def solve_distinct_disks(length, n):
    pass

############################################################
# Section 4: Feedback
############################################################

feedback_question_1 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
