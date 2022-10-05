############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Haorui Lyu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

import copy
import random
from queue import PriorityQueue

############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    '''
    >>> p = create_tile_puzzle(3, 3)
    >>> p.get_board()
    [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    >>> p = create_tile_puzzle(2, 4)
    >>> p.get_board()
    [[1, 2, 3, 4], [5, 6, 7, 0]]
    '''
    board = [[0] *cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            board[i][j] = 1 + (i * cols) + j
    board[rows-1][cols-1] = 0
    return TilePuzzle(board)

class TilePuzzle(object):
    # Required
    def __init__(self, board):
        self.direction = ['up', 'down', 'left', 'right']
        self.board = board
        self.n = len(board)
        self.m = len(board[0])
        for i in range(self.n):
            for j in range(self.m):
                if self.board[i][j] == 0:
                    self.x = i
                    self.y = j
                    return

    def get_board(self):
        '''
        >>> p = TilePuzzle([[1, 2], [3, 0]])
        >>> p.get_board()
        [[1, 2], [3, 0]]
        >>> p = TilePuzzle([[0, 1], [3, 2]])
        >>> p.get_board()
        [[0, 1], [3, 2]]
        '''
        return self.board

    def perform_move(self, direction):
        '''
        >>> p = create_tile_puzzle(3, 3)
        >>> p.perform_move("up")
        True
        >>> p.get_board()
        [[1, 2, 3], [4, 5, 0], [7, 8, 6]]
        >>> p = create_tile_puzzle(3, 3)
        >>> p.perform_move("down")
        False
        >>> p.get_board()
        [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        '''
        i,j = self.x,self.y
        if direction == 'up':i -= 1
        elif direction == 'down':i += 1
        elif direction == 'left':j -= 1
        elif direction == 'right':j += 1
        else:return False
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            return False
        self.board[i][j],self.board[self.x][self.y] = \
            self.board[self.x][self.y],self.board[i][j]
        self.x,self.y = i,j
        return True

    def scramble(self, num_moves):
        for i in range(num_moves):
            self.perform_move(random.choice(self.direction))

    def is_solved(self):
        '''
        >>> p = TilePuzzle([[1, 2], [3, 0]])
        >>> p.is_solved()
        True
        >>> p = TilePuzzle([[0, 1], [3, 2]])
        >>> p.is_solved()
        False
        '''
        for i in range(self.n):
            for j in range(self.m-1):
                if self.board[i][j] != 1 + i * self.m + j:
                    return False
        return self.board[self.n-1][self.m-1] == 0

    def copy(self):
        '''
        >>> p = create_tile_puzzle(3, 3)
        >>> p2 = p.copy()
        >>> p.get_board() == p2.get_board()
        True
        >>> p = create_tile_puzzle(3, 3)
        >>> p2 = p.copy()
        >>> p.perform_move("left")
        True
        >>> p.get_board() == p2.get_board()
        False
        '''
        return copy.deepcopy(self)

    def successors(self):
        '''
        >>> p = create_tile_puzzle(3, 3)
        >>> for move, new_p in p.successors():
        ...     print (move, new_p.get_board())
        up [[1, 2, 3], [4, 5, 0], [7, 8, 6]]
        left [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
        >>> b = [[1,2,3], [4,0,5], [6,7,8]]
        >>> p = TilePuzzle(b)
        >>> for move, new_p in p.successors():
        ...     print (move, new_p.get_board())
        up [[1, 0, 3], [4, 2, 5], [6, 7, 8]]
        down [[1, 2, 3], [4, 7, 5], [6, 0, 8]]
        left [[1, 2, 3], [0, 4, 5], [6, 7, 8]]
        right [[1, 2, 3], [4, 5, 0], [6, 7, 8]]
        '''
        for d in self.direction:
            newobj = self.copy()
            if(newobj.perform_move(d)):
                yield (d,newobj)

    def iddfs_helper(self, limit, moves):
        if self.is_solved():
            yield moves
        if limit == 0:return
        for i in range(len(self.direction)):
            if self.perform_move(self.direction[i]):
                yield from self.iddfs_helper(limit-1,moves + [self.direction[i]])
                self.perform_move(self.direction[i^1])

    # Required
    def find_solutions_iddfs(self):
        '''
        >>> b = [[4,1,2], [0,5,3], [7,8,6]]
        >>> p = TilePuzzle(b)
        >>> solutions = p.find_solutions_iddfs()
        >>> next(solutions)
        ['up', 'right', 'right', 'down', 'down']
        >>> b = [[1,2,3], [4,0,8], [7,6,5]]
        >>> p = TilePuzzle(b)
        >>> list(p.find_solutions_iddfs())
        [['down', 'right', 'up', 'left', 'down', 'right'], ['right', 'down', 'left', 'up', 'right', 'down']]
        '''
        # return self.iddfs_helper(6,[])
        limit = 0
        while True:
            limit += 1
            ans = self.iddfs_helper(limit,[])
            if list(ans) != []:
                return self.iddfs_helper(limit,[])
    # Required
    def hs(self):
        hscode = ""
        for i in range(self.n):
            for j in range(self.m):
                hscode += str(self.board[i][j])
        return hscode

    def H(self):
        h = 0
        for i in range(self.n):
            for j in range(self.m):
                x = self.board[i][j]
                if x == 0:
                    h += abs(i - self.n + 1) + abs(j - self.m + 1)
                else:
                    h += abs(i - ((x-1)//self.m)) + abs(j - ((x-1)%self.m))
        return h
    class node:
        def __init__(self,obj,moves,dis):
            self.obj = obj
            self.moves = moves
            self.dis = dis

        def __lt__(self, other):
            return self.dis < other.dis
    def find_solution_a_star(self):
        '''
        >>> b = [[4,1,2], [0,5,3], [7,8,6]]
        >>> p = TilePuzzle(b)
        >>> p.find_solution_a_star()
        ['up', 'right', 'right', 'down', 'down']
        >>> b = [[1,2,3], [4,0,5], [6,7,8]]
        >>> p = TilePuzzle(b)
        >>> p.find_solution_a_star()
        ['right', 'down', 'left', 'left', 'up', 'right', 'down', 'right', 'up', 'left', 'left', 'down', 'right', 'right']
        '''
        visited = set()
        pq = PriorityQueue()
        pq.put(self.node(self,[],0))
        while not pq.empty():
            cur = pq.get()
            obj,moves,dis = cur.obj,cur.moves,cur.dis
            if obj.hs() in visited:continue
            visited.add(obj.hs())
            if obj.H() == 0:
                return moves
            for mv,nx_obj in obj.successors():
                h = nx_obj.H()
                pq.put(self.node(nx_obj,moves + [mv],dis + h))
        return -1

############################################################
# Section 2: Grid Navigation
############################################################

class node:
    def __init__(self,pos,moves,dis):
        self.pos = pos
        self.moves = moves
        self.dis = dis

    def __lt__(self, other):
        return self.dis < other.dis
def H(cur,goal):
    return ((cur[0] - goal[0]) ** 2 + (cur[1] - goal[1]) ** 2) ** 0.5

def find_path(start, goal, scene):
    '''
    >>> scene = [[False, False, False],
    ...         [False, True , False],
    ...         [False, False, False]]
    >>> find_path((0, 0), (2, 1), scene)
    [(0, 0), (1, 0), (2, 1)]
    >>> scene = [[False, True, False],
    ...         [False, True, False],
    ...         [False, True, False]]
    >>> print(find_path((0, 0), (0, 2), scene))
    None
    '''
    if scene[start[0]][start[1]] == True or\
        scene[goal[0]][goal[1]] == True:
        return None
    pq = PriorityQueue()
    pq.put(node(start,[start],0))
    visited = set()
    while not pq.empty():
        cur = pq.get()
        pos,moves,dis = cur.pos,cur.moves,cur.dis
        if pos in visited:continue
        visited.add(pos)
        if pos == goal:
            return moves
        for dx in range(-1,2):
            for dy in range(-1,2):
                if dx == 0 and dy == 0:continue
                nx = pos[0] + dx
                ny = pos[1] + dy
                if nx < 0 or nx >= len(scene) or ny < 0 or \
                        ny >= len(scene[0]) or scene[nx][ny] == True:
                    continue
                new_dis = dis + (dx**2 + dy**2)**0.5 + H(pos,goal)
                pq.put(node((nx,ny),moves + [(nx,ny)],new_dis))
    return None

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

class node3:
    def __init__(self,st,moves,dis):
        self.st = st
        self.moves = moves
        self.dis = dis

    def __lt__(self, other):
        return self.dis < other.dis
def hs3(cur):
    hs = ""
    for c in cur:
        hs += str(c)
    return hs
def H3(cur,length):
    ans = 0
    for i in range(length):
        if cur[i] == 0:continue
        ans += abs(i - (length - cur[i]))
    return ans
def solve_distinct_disks(length, n):
    '''
    >>> solve_distinct_disks(5,2)
    [(0, 2), (1, 3), (2, 4)]
    >>> solve_distinct_disks(4,3)
    [(1, 3), (0, 1), (2, 0), (3, 2), (1, 3), (0, 1)]
    '''
    pq = PriorityQueue()
    start = [0] * length
    for i in range(n):
        start[i] = i+1
    goal = [0] * length
    id = 1
    for i in range(length-1,length-n-1,-1):
        goal[i] = id
        id += 1
    pq.put(node3(start,[],0))
    visited = set()
    while not pq.empty():
        cur = pq.get()
        st,moves,dis = cur.st,cur.moves,cur.dis
        if hs3(st) in visited:continue
        visited.add(hs3(st))
        if st == goal:
            return moves
        for i in range(length):
            if st[i] != 0:
                if i + 1 < length and st[i + 1] == 0:
                    st2 = copy.deepcopy(st)
                    st2[i],st2[i+1] = st2[i+1],st2[i]
                    new_moves = moves + [(i,i+1)]
                    new_dis = dis + 1 + H3(st2,length)
                    pq.put(node3(st2,new_moves,new_dis))
                if i - 1 >=0 and st[i - 1] == 0:
                    st2 = copy.deepcopy(st)
                    st2[i], st2[i - 1] = st2[i - 1], st2[i]
                    new_moves = moves + [(i, i - 1)]
                    new_dis = dis + 1 + H3(st2, length)
                    pq.put(node3(st2, new_moves, new_dis))
                if i + 2 < length and st[i + 1] != 0 and st[i+2] == 0:
                    st2 = copy.deepcopy(st)
                    st2[i], st2[i + 2] = st2[i + 2], st2[i]
                    new_moves = moves + [(i, i + 2)]
                    new_dis = dis + 1 + H3(st2, length)
                    pq.put(node3(st2, new_moves, new_dis))
                if i - 2 >= 0 and st[i - 1] != 0 and st[i-2] == 0:
                    st2 = copy.deepcopy(st)
                    st2[i], st2[i - 2] = st2[i - 2], st2[i]
                    new_moves = moves + [(i, i - 2)]
                    new_dis = dis + 1 + H3(st2, length)
                    pq.put(node3(st2, new_moves, new_dis))

############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    '''
    >>> g = create_dominoes_game(2, 2)
    >>> g.get_board()
    [[False, False], [False, False]]
    >>> g = create_dominoes_game(2, 3)
    >>> g.get_board()
    [[False, False, False], [False, False, False]]
    '''
    board = [[False] * cols for _ in range(rows)]
    return DominoesGame(board)

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.m = len(board[0])

    def get_board(self):
        '''
        >>> b = [[False, False], [False, False]]
        >>> g = DominoesGame(b)
        >>> g.get_board()
        [[False, False], [False, False]]
        >>> b = [[True, False], [True, False]]
        >>> g = DominoesGame(b)
        >>> g.get_board()
        [[True, False], [True, False]]
        '''
        return self.board

    def reset(self):
        '''
        >>> b = [[False, False], [False, False]]
        >>> g = DominoesGame(b)
        >>> g.get_board()
        [[False, False], [False, False]]
        >>> g.reset()
        >>> g.get_board()
        [[False, False], [False, False]]
        >>> b = [[True, False], [True, False]]
        >>> g = DominoesGame(b)
        >>> g.get_board()
        [[True, False], [True, False]]
        >>> g.reset()
        >>> g.get_board()
        [[False, False], [False, False]]
        '''
        for i in range(self.n):
            for j in range(self.m):
                self.board[i][j] = False

    def is_legal_move(self, row, col, vertical):
        '''
        >>> b = [[False, False], [False, False]]
        >>> g = DominoesGame(b)
        >>> g.is_legal_move(0, 0, True)
        True
        >>> g.is_legal_move(0, 0, False)
        True
        >>> b = [[True, False], [True, False]]
        >>> g = DominoesGame(b)
        >>> g.is_legal_move(0, 0, False)
        False
        >>> g.is_legal_move(0, 1, True)
        True
        >>> g.is_legal_move(1, 1, True)
        False
        '''
        row2,col2 = row,col
        if vertical:row2 += 1
        else: col2 += 1
        if row <0 or row >= self.n or col < 0 or col >= self.m:
            return False
        if row2 <0 or row2 >= self.n or col2 < 0 or col2 >= self.m:
            return False
        return self.board[row][col] == False and self.board[row2][col2] == False

    def legal_moves(self, vertical):
        '''
        >>> g = create_dominoes_game(3, 3)
        >>> list(g.legal_moves(True))
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> list(g.legal_moves(False))
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        >>> b = [[True, False], [True, False]]
        >>> g = DominoesGame(b)
        >>> list(g.legal_moves(True))
        [(0, 1)]
        >>> list(g.legal_moves(False))
        []
        '''
        for i in range(self.n):
            for j in range(self.m):
                x1,y1 = i,j
                if vertical:x2,y2 = i + 1,j
                else: x2,y2 = i,j+1
                if x2 <0 or x2 >= self.n or y2 < 0 or y2 >= self.m:
                    continue
                if self.board[x1][y1] == False and self.board[x2][y2] == False:
                    yield (x1,y1)

    def perform_move(self, row, col, vertical):
        '''
        >>> g = create_dominoes_game(3, 3)
        >>> g.perform_move(0, 1, True)
        >>> g.get_board()
        [[False, True, False], [False, True, False], [False, False, False]]
        >>> g = create_dominoes_game(3, 3)
        >>> g.perform_move(1, 0, False)
        >>> g.get_board()
        [[False, False, False], [True, True, False], [False, False, False]]
        '''
        self.board[row][col] = True
        if vertical:
            self.board[row+1][col] = True
        else:
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        '''
        >>> b = [[False, False], [False, False]]
        >>> g = DominoesGame(b)
        >>> g.game_over(True)
        False
        >>> g.game_over(False)
        False
        '''
        return list(self.legal_moves(vertical)) == []


    def copy(self):
        '''
        >>> g = create_dominoes_game(4, 4)
        >>> g2 = g.copy()
        >>> g.get_board() == g2.get_board()
        True
        >>> g = create_dominoes_game(4, 4)
        >>> g2 = g.copy()
        >>> g.perform_move(0, 0, True)
        >>> g.get_board() == g2.get_board()
        False
        '''
        return copy.deepcopy(self)

    def successors(self, vertical):
        '''
        >>> b = [[False, False], [False, False]]
        >>> g = DominoesGame(b)
        >>> for m, new_g in g.successors(True):
        ...     print(m, new_g.get_board())
        (0, 0) [[True, False], [True, False]]
        (0, 1) [[False, True], [False, True]]
        >>> b = [[True, False], [True, False]]
        >>> g = DominoesGame(b)
        >>> for m, new_g in g.successors(True):
        ...     print(m, new_g.get_board())
        (0, 1) [[True, True], [True, True]]
        '''
        for pos in self.legal_moves(vertical):
            newobj = self.copy()
            newobj.perform_move(pos[0],pos[1],vertical)
            yield (pos,newobj)

    def get_random_move(self, vertical):
        next_steps = list(self.legal_moves(vertical))
        move = random.choice(next_steps)
        self.perform_move(move[0],move[1],vertical)

    nodes = 0
            
    # Required
    def get_best_move(self, vertical, limit):
        '''
        >>> b = [[False] * 3 for i in range(3)] 
        >>> g = DominoesGame(b) 
        >>> g.get_best_move(True, 1) 
        ((0, 1), 2, 6) 
        >>> g.get_best_move(True, 2) 
        ((0, 1), 3, 10) 
        >>> b = [[False] * 3 for i in range(3)] 
        >>> g = DominoesGame(b) 
        >>> g.perform_move(0, 1, True) 
        >>> g.get_best_move(False, 1) 
        ((2, 0), -3, 2) 
        >>> g.get_best_move(False, 2) 
        ((2, 0), -2, 5) 
        '''
        move, value = self.alphabeta(limit, float("-inf"), float("inf"), vertical, vertical, True)
        temp = DominoesGame.nodes
        DominoesGame.nodes = 0
        return move, value, temp
    
    def alphabeta(self, depth, alpha, beta, vertical, root, maximizingPlayer):
        if depth == 0 or self.game_over(vertical):
            DominoesGame.nodes += 1
            return ((0,0), len(list(self.legal_moves(root))) - len(list(self.legal_moves(not root))))
        if maximizingPlayer:
            v = float("-inf")
            need_move = tuple()
            for move, board in self.successors(vertical):
                state, temp = board.alphabeta(depth - 1, alpha, beta, not vertical, root, False)
                if temp > v:
                    v = temp
                    need_move = move
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return (need_move, v)
        else:
            v = float("inf")
            need_move = tuple()
            for move, board in self.successors(vertical):
                state, temp = board.alphabeta(depth - 1, alpha, beta, not vertical, root, True)
                if temp < v:
                    v = temp
                    need_move = move
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return (need_move, v)
        
        
############################################################
# Section 5: Feedback
############################################################

feedback_question_1 = """
I took about 20 hours.
"""

feedback_question_2 = """
I think the alpha-beta search algorithm is most challenging。 
At first I don't know how to implement this algorithem. I find a lot of video online,
then implement this algorithem follow the pseudo-code on the Wikipedia.
"""

feedback_question_3 = """
I love all search method we use in this assignment. That is very intresting.
"""
