############################################################
# CMPSC 442: Homework 1
############################################################

student_name = "Haorui Lyu"

############################################################
# Section 1: Working with Lists
############################################################

def ejtract_and_apply(l, p, f):
    return [f(j) for j in l if p(j)]

def concatenate(seqs):
    return [j for e in seqs for j in e]

def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))]for j in range(len(matrix[0]))]

############################################################
# Section 2: Sequence Slicing
############################################################

def copy(seq):
    return seq[:]

def all_but_last(seq):
    return seq[:-1]

def every_other(seq):
    return seq[::2]

############################################################
# Section 3: Combinatorial Algorithms
############################################################

def prefixes(seq):
    for j in range(len(seq) + 1):
        yield seq[:j]

def suffixes(seq):
    for j in range(len(seq) + 1):
        yield seq[j:]


def slices(seq):
    for i in range(0, len(seq)):
        for j in range(i + 1, len(seq) + 1):
            yield seq[i:j]

############################################################
# Section 4: Text Processing
############################################################

def normalize(text):
    text = text.lower()
    text = text.split()
    return " ".join(text)

def no_vowels(text):
    y = []
    for j in text:
        if j not in "aeiouAEIOU":
            y.append(j)
    return "".join(y)

def digits_to_words(text):
    numbers_in_words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                        "6": "six", "7": "seven", "8": "eight", "9": "nine" }
    output_put = []
    for i in text:
        if i in numbers_in_words:
            output_put.append(numbers_in_words[i])
    return " ".join(output_put)

def to_mixed_case(name):
    name = name.replace("_", " ").lower().split()
    name[1:] = map(lambda j: j[0].upper() + j[1:], name[1:])
    return "".join(name)

############################################################
# Section 5: Polynomials
############################################################

class Polynomial(object):

    def __init__(self, polynomial):
        self.polynomial = tuple(polynomial)

    def get_polynomial(self):
        return self.polynomial

    def __neg__(self):
        return Polynomial((-j[0], j[1]) for j in self.polynomial)

    def __add__(self, other):
        return Polynomial(self.polynomial + other.get_polynomial())

    def __sub__(self, other):
        return Polynomial(self.polynomial + (-other).get_polynomial())

    def __mul__(self, other):
        final_out = []
        for i in self.polynomial:
            for j in other.polynomial:
                final_out.append((i[0] * j[0],i[1] + j[1]))
        return Polynomial(final_out)


    def __call__(self, j):
        return sum((i[0] * j ** i[1]) for i in self.polynomial)


    def simplify(self):
        mp = {}
        for t in self.polynomial:
            if not mp.get(t[1]):
                mp[t[1]] = 0
            mp[t[1]] += t[0]
        self.polynomial = []
        for p,v in mp.items():
            if v != 0:
                self.polynomial.append((v,p))
        self.polynomial.sort(key=lambda x:-x[1])
        if self.polynomial == []:
            self.polynomial = [(0,0)]
        self.polynomial = tuple(self.polynomial)

    def __str__(self):
        ans = ""
        for i in range(len(self.polynomial)):
            a,p = self.polynomial[i][0],self.polynomial[i][1]
            if i == 0:
                if a < 0:ans += "-"
            else:
                if a >= 0:ans += " + "
                else:ans += " - "
            if p > 0:
                if abs(a) != 1:ans += str(abs(a)) + "x"
                else: ans += "x"
                if p != 1: ans += "^" + str(p)
            else:
                ans += str(abs(a))
        return ans


############################################################
# Section 6: Feedback
############################################################

feedback_question_1 = """
I use about 7.5 hours.
"""

feedback_question_2 = """
I think it's the polynomial part. Especially the str and simplify functions.
"""

feedback_question_3 = """
Sequence Slicing part. After that I can write cleaner and more readable code.
"""
