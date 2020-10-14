"""
Lior Sinai, 9 October 2020

Integer partitions

See:
https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
https://www.geeksforgeeks.org/generate-unique-partitions-of-an-integer/
https://stackoverflow.com/questions/17720072/print-all-unique-integer-partitions-given-an-integer-as-input

"""
from itertools import permutations

def integer_partitions2(n, parts, k=0, min_val=1):
    # unique partitions with passing a list that may have inital values
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


def unique_perm_partitions(n, k):
    # get unique permutations. There are cleverer but more complex ways
    # https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    for partition in integer_partitions(n, k): # unique partitions
        for perm in set(permutations(partition)): # generate all permutations and find the set
            yield perm


def integer_partitions(n, k, min_val=0):
    # unique partitions with no list
    if n < min_val:
        return
    if k == 1:
        yield (n, )
    else:
        for i in range(min_val, n + 1):
            for result in integer_partitions(n-i, k - 1, min_val=i):
                yield (i,) + result 

# def integer_parts(n, parts):
#    # has duplicates
#     if n == 0:
#         yield parts
#     else:
#         for i in range(0, len(parts)):
#             parts_next = parts[:]
#             parts_next[i] += 1
#             yield from integer_parts(n-1, parts_next)





if __name__ == '__main__':
    #n, parts = 5, [0] * 3
    #n, parts = 6, [0] * 3

    n, parts = 10, [0] * 3

    # for partition in integer_partitions(n, len(parts)):
    #     print(partition)

    # n, parts = 10, [0, 0, 0]
    for i, partition in enumerate(unique_perm_partitions(n, len(parts))):
        print(i, partition)

    # for p1, p2 in zip(integer_partitions_nk(n, parts), integer_partitions_nk(n, len(parts))):
    #     print(p1, p2)

    run = (3, 3, 1)
    sequence = [3, 3, 1, 3, 3, 1, 1, 1, 3, 3, 3, 3, 1, 3, 3]
    possible = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
    possible = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]