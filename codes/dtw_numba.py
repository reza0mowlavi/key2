import numpy as np
import numba


@numba.njit(nogil=True, fastmath=True)
def minimum3(x, y, z):
    min_ = x if x <= y else y
    min_ = min_ if min_ <= z else z
    return min_


@numba.njit(nogil=True, fastmath=True)
def abstract(x):
    if x >= 0:
        return x
    else:
        return -x


@numba.njit(nogil=True, fastmath=True)
def squared_dist_scalar(x, y, squared=True):
    dist = x - y
    if squared:
        return dist * dist
    else:
        return abstract(dist)


@numba.njit(nogil=True, fastmath=True)
def squared_dist_multidim(x, y, squared=True):
    s = 0
    for i in range(len(x)):
        s += squared_dist_scalar(x[i], y[i], squared=squared)

    return s


@numba.njit(nogil=True, fastmath=True)
def squared_dist(x, y, squared=True):
    if hasattr(x, "shape"):
        return squared_dist_multidim(x, y, squared)
    else:
        return squared_dist_scalar(x, y, squared)


@numba.njit(nogil=True, parallel=True)
def padded_parallel_pairwise_distance(X, lens, squared=True):
    N = len(X)
    M = int(N * (N - 1) / 2)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    rows = np.empty((M,), dtype="int32")
    cols = np.empty((M,), dtype="int32")

    counter = 0
    for row in range(1, N):
        for col in range(row):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(M):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = result[col, row] = dtw(
            X[row, : lens[row]], X[col, : lens[col]], squared=squared
        )

    return result


@numba.njit(nogil=True, parallel=True)
def parallel_pairwise_distance(X, squared=True):
    N = len(X)
    M = int(N * (N - 1) / 2)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    rows = np.empty((M,), dtype="int32")
    cols = np.empty((M,), dtype="int32")

    counter = 0
    for row in range(1, N):
        for col in range(row):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(M):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = result[col, row] = dtw(X[row], X[col], squared=squared)

    return result


@numba.njit(nogil=True, parallel=True)
def padded_parallel_batch_distance(X, Y, X_lens, Y_lens, squared=True):
    N = len(X)
    M = len(Y)
    ALL = M * N
    result = np.empty((N, M), dtype=X.dtype)

    rows = np.empty((ALL,), dtype="int32")
    cols = np.empty((ALL,), dtype="int32")

    counter = 0
    for row in range(N):
        for col in range(M):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(ALL):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = dtw(
            X[row, : X_lens[row]], Y[col, : Y_lens[col]], squared=squared
        )

    return result


@numba.njit(nogil=True, parallel=True)
def parallel_batch_distance(X, Y, squared=True):
    N = len(X)
    M = len(Y)
    ALL = M * N
    result = np.empty((N, M), dtype=X.dtype)

    rows = np.empty((ALL,), dtype="int32")
    cols = np.empty((ALL,), dtype="int32")

    counter = 0
    for row in range(N):
        for col in range(M):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(ALL):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = dtw(X[row], Y[col], squared=squared)

    return result


@numba.njit(nogil=True)
def batch_distance(X, Y, squared=True):
    N = len(X)
    M = len(Y)
    result = np.empty((N, M), dtype=X.dtype)

    for row in range(N):
        for col in range(M):
            result[row, col] = dtw(X[row], Y[col], squared=squared)

    return result


@numba.njit(nogil=True)
def padded_batch_distance(X, Y, X_lens, Y_lens, squared=True):
    N = len(X)
    M = len(Y)
    result = np.empty((N, M), dtype=X.dtype)

    for row in range(N):
        for col in range(M):
            result[row, col] = dtw(
                X[row, : X_lens[row]], Y[col, : Y_lens[col]], squared=squared
            )

    return result


@numba.njit(nogil=True)
def padded_pairwise_distance(X, lens, squared=True):
    N = len(X)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    for row in range(1, N):
        for col in range(row):
            result[row, col] = result[col, row] = dtw(
                X[row, : lens[row]], X[col, : lens[col]], squared=squared
            )

    return result


@numba.njit(nogil=True)
def pairwise_distance(X, squared=True):
    N = len(X)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    for row in range(1, N):
        for col in range(row):
            result[row, col] = result[col, row] = dtw(X[row], X[col], squared=squared)

    return result


@numba.njit(nogil=True, fastmath=True)
def dtw_matrix(seq1, seq2, squared=True):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    # Create a 2D array to store the accumulated distances
    dtw_matrix = np.empty((len_seq1 + 1, len_seq2 + 1), dtype=seq1.dtype)

    # Initialize the first row and column of the matrix
    for i in range(1, len_seq1 + 1):
        dtw_matrix[i, 0] = np.finfo(seq1.dtype).max

    for j in range(1, len_seq2 + 1):
        dtw_matrix[0, j] = np.finfo(seq1.dtype).max

    dtw_matrix[0, 0] = 0

    # Fill in the rest of the matrix
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            dtw_matrix[i, j] = squared_dist(
                seq1[i - 1], seq2[j - 1], squared=squared
            ) + minimum3(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
            )

    # The DTW distance is the value in the bottom-right corner of the matrix

    return dtw_matrix


@numba.njit(nogil=True)
def dtw(seq1, seq2, squared=True):
    return dtw_matrix(seq1, seq2, squared=squared)[-1, -1]


@numba.njit(nogil=True, fastmath=True)
def maximum3(x, y, z):
    max_ = x if x >= y else y
    max_ = max_ if max_ >= z else z
    return max_


@numba.njit(nogil=True, fastmath=True)
def softmin3(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = maximum3(a, b, c)

    tmp = 0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)
    softmin_value = -gamma * (np.log(tmp) + max_val)
    return softmin_value


@numba.njit(nogil=True, fastmath=True)
def soft_dtw_matrix(seq1, seq2, gamma=0.1, squared=True):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    # Create a 2D array to store the accumulated distances
    dtw_matrix = np.empty((len_seq1 + 1, len_seq2 + 1), dtype=seq1.dtype)

    # Initialize the first row and column of the matrix
    for i in range(1, len_seq1 + 1):
        dtw_matrix[i, 0] = np.finfo(seq1.dtype).max

    for j in range(1, len_seq2 + 1):
        dtw_matrix[0, j] = np.finfo(seq1.dtype).max

    dtw_matrix[0, 0] = 0

    # Fill in the rest of the matrix
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            dtw_matrix[i, j] = squared_dist(
                seq1[i - 1], seq2[j - 1], squared=squared
            ) + softmin3(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
                gamma=gamma,
            )

    # The DTW distance is the value in the bottom-right corner of the matrix

    return dtw_matrix


@numba.njit(nogil=True)
def soft_dtw(seq1, seq2, gamma=0.1, squared=True):
    return soft_dtw_matrix(seq1, seq2, gamma=gamma, squared=squared)[-1, -1]


@numba.njit(nogil=True, parallel=True)
def soft_parallel_pairwise_distance(X, gamma, squared=True):
    N = len(X)
    M = int(N * (N - 1) / 2)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    rows = np.empty((M,), dtype="int32")
    cols = np.empty((M,), dtype="int32")

    counter = 0
    for row in range(1, N):
        for col in range(row):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(M):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = result[col, row] = soft_dtw(
            X[row, : lens[row]], X[col, : lens[col]], gamma=gamma, squared=squared
        )

    return result


@numba.njit(nogil=True, parallel=True)
def soft_padded_parallel_pairwise_distance(X, gamma, lens, squared=True):
    N = len(X)
    M = int(N * (N - 1) / 2)

    result = np.empty((N, N), dtype=X.dtype)
    for d in range(N):
        result[d, d] = 0

    rows = np.empty((M,), dtype="int32")
    cols = np.empty((M,), dtype="int32")

    counter = 0
    for row in range(1, N):
        for col in range(row):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(M):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = result[col, row] = soft_dtw(
            X[row], X[col], gamma=gamma, squared=squared
        )

    return result


@numba.njit(nogil=True, parallel=True)
def soft_parallel_batch_distance(X, Y, gamma, squared=True):
    N = len(X)
    M = len(Y)
    ALL = M * N
    result = np.empty((N, M), dtype=X.dtype)

    rows = np.empty((ALL,), dtype="int32")
    cols = np.empty((ALL,), dtype="int32")

    counter = 0
    for row in range(N):
        for col in range(M):
            rows[counter] = row
            cols[counter] = col
            counter += 1

    for counter in numba.prange(ALL):
        row = rows[counter]
        col = cols[counter]
        result[row, col] = soft_dtw(X[row], Y[col], gamma=gamma, squared=squared)

    return result


def alignment_from_dtw_matrix(dtw_matrix):
    n, m = dtw_matrix.shape
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )
            if min_cost == dtw_matrix[i - 1, j]:
                i -= 1
            elif min_cost == dtw_matrix[i, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    path.reverse()
    return path
