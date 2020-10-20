"""
Lior Sinai, 9 October 2020

Integer partitions

See:
https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
https://www.geeksforgeeks.org/generate-unique-partitions-of-an-integer/
https://stackoverflow.com/questions/17720072/print-all-unique-integer-partitions-given-an-integer-as-input

"""
from itertools import permutations

def integer_partitions2(n, parts, k=0, min_val=0):
    # unique partitions with passing a list that may have initial values
    if n < min_val:
        return
    if k == len(parts) - 1:
        parts[k] = n
        yield parts
    else:
        for i in range(max(min_val, parts[k]), n + 1):
            parts_next = parts[:]
            parts_next[k] = i
            yield from integer_partitions2(n-i, parts_next, k + 1, min_val=i)


class UniqueElement:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    # answer to https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    eset = set(elements)
    listunique = [UniqueElement(i,elements.count(i)) for i in eset]
    u = len(elements)
    return perm_unique_helper(listunique, [0]*u, u-1)

def perm_unique_helper(listunique, result_list, d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d] = i.value
                i.occurrences -= 1
                for g in  perm_unique_helper(listunique, result_list, d-1):
                    yield g
                i.occurrences += 1

def unique_perm_partitions(n, k):
    """ get unique permutations of integer partitions"""
    for partition in integer_partitions(n, k): # unique partitions
        for perm in perm_unique(partition): #set(permutations(partition)): # generate all permutations and find the set
            yield perm


def integer_partitions(n:int, k:int, min_val=0):
    """ unique partitions of length k for an integer n"""
    if n < min_val:
        return
    if k == 1:
        yield (n, )
    else:
        for i in range(min_val, n + 1):
            for result in integer_partitions(n-i, k - 1, min_val=i):
                yield (i,) + result 
                

if __name__ == '__main__':
    #n, parts = 5, [0] * 3
    #n, parts = 6, [0] * 3

    n, parts = 10, [0] * 3

    # for partition in integer_partitions(n, len(parts)):
    #     print(partition)

    # n, parts = 10, [0, 0, 0]
    for i, partition in enumerate(unique_perm_partitions(n, len(parts))):
        print(i, partition)

    print("")
    for i, p in enumerate(zip(integer_partitions2(n, parts), integer_partitions(n, len(parts)))):
        print(i, p[0], p[1])