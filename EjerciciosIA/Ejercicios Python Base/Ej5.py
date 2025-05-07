
def sumTuples (tuples):
    result = []
    for t in tuples:
        sum = 0
        for i in range(len(t)):
            sum += t[i]
        result.append(sum)
    result.sort(reverse=True)
    return result

tuplas = [(3,-2), (0,5,3), (-1,-4,-4), (2,-3), (-5,0,-2), (4,-5)]
print(sumTuples(tuplas))
