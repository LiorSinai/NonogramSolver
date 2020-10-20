BLACK = 1   # = 01 in binary
WHITE = 2   # = 10 in binary
EITHER = 3  # = 11 in binary

class Match():
    def __init__(self, match=None, pattern=None, span=None):
        self.match = match
        self.pattern = pattern
        self.span = span

    @property
    def is_match(self):
        return self.match is not None


def minimum_sequence(pattern, length):
    """ Returns the minimum sequence of a pattern, which is one WHITE between every black"""
    match = []
    for n_black in pattern:
        match += [WHITE] + [BLACK]  * n_black
    match = match[1:] # chop off the first WHITE
    match += [WHITE] * (length - len(match)) 
    return match


def special_matches(array, pattern):
    # special case optimisation
    if not pattern:
        # match an empty sequence
        if array.count(BLACK) == 0:
            match = [WHITE] * len(array)
            return Match(match, pattern=pattern)     
        else:
            return Match(pattern=pattern) #no match
    if array.count(BLACK) == 0 and array.count(WHITE) ==0:
        # construct minimum pattern
        return Match(minimum_sequence(pattern, len(array)), pattern=pattern)
    else:
        return Match(pattern=pattern)  # no match
