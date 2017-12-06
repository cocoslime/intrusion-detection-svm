def innerIntersect(A, B):
    result = []
    for i in range(0, len(A)):
        set_a = set(A[i]);
        for j in range(0, len(B)):
            set_b = set(B[j])
            set_result = set_a.intersection(set_b)
            if len(set_result) != 0 :
                result.append(list(set_result))
    return result;