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


def listRightIndex(array, value):
    """Returns the index for the right-most matching value"""
    return len(array) - array[-1::-1].index(value) -1


def special_matches(array, pattern):
    # special case optimisation
    count_b = array.count(BLACK)
    count_w = array.count(WHITE) 
    if not pattern:
        # match an empty sequence
        if count_b == 0:
            match = [WHITE] * len(array)
            return Match(match, pattern=pattern)     
        else:
            return Match(pattern=pattern) #no match
    elif count_b == 0 and count_w ==0:
        # construct minimum pattern
        return Match(minimum_sequence(pattern, len(array)), pattern=pattern)
    elif count_b > 0 and count_w ==0:
        # check for near worst case, where have nothing but the last part of the sequence is on the other end
        min_length = sum(pattern) + (len(pattern)-1) # 1 white interval
        idx1 = array.index(BLACK)
        idx2 = listRightIndex(array, BLACK)
        if idx1 > min_length and idx2 - idx1 < pattern[-1] and (idx2+1)-pattern[-1] > min_length:
            m = minimum_sequence(pattern[:-1], (idx2+1)-pattern[-1])
            m += [BLACK] * pattern[-1]
            m += [WHITE] * (len(array) - len(m))
            return Match(m, pattern=pattern)
        else:
            return Match(pattern=pattern)  # no match
    else:
        return Match(pattern=pattern)  # no match
