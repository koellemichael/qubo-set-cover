import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler

def matrix_to_dict(matrix):
    res_dict = {}
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            res_dict[(i, j)] = matrix[i][j]
    return res_dict


def set_cover_qubo(u, v, alpha=2, beta=1):
    # construct matrix with ancilla qubits
    ancilla = []
    for a in range(len(u)):
        count = 0
        for m in range(len(v)):
            if u[a] in v[m]:
                count += 1
                ancilla.append((u[a], count))

    qubo_matrix = np.zeros((len(ancilla) + len(v), len(ancilla) + len(v)))

    # penalty 1
    for i in range(len(ancilla)):
        (a_i, m_i) = ancilla[i]
        for j in range(len(ancilla)):
            (a_j, m_j) = ancilla[j]
            if i == j:
                qubo_matrix[i][j] += alpha * -1
            elif a_i == a_j:
                qubo_matrix[i][j] += alpha * 2

    # penalty 2
    for i in range(len(ancilla)):
        (a_i, m_i) = ancilla[i]
        for j in range(len(ancilla)):
            (a_j, m_j) = ancilla[j]
            if a_i == a_j:
                if i == j:
                    qubo_matrix[i][j] += alpha * m_i * m_j * 1
                else:
                    qubo_matrix[i][j] += alpha * m_i * m_j * 2

    for x_j in range(len(v)):
        for i in range(len(ancilla)):
            (a_i, m_i) = ancilla[i]
            if a_i in v[x_j]:
                qubo_matrix[i][len(ancilla) + x_j] += alpha * m_i * -2

    for a in u:
        for i in range(len(v)):
            for j in range(len(v)):
                if a in v[i] and a in v[j]:
                    if i == j:
                        qubo_matrix[len(ancilla) + i][len(ancilla) + j] += alpha * 1
                    else:
                        qubo_matrix[len(ancilla) + i][len(ancilla) + j] += alpha * 2

    # cost function
    for x_i in range(len(v)):
        for x_j in range(len(v)):
            if x_i == x_j:
                qubo_matrix[x_i + len(ancilla)][x_j + len(ancilla)] += beta * 1

    # lower triangle to zero
    for x_i in range(len(v) + len(ancilla)):
        for x_j in range(len(v) + len(ancilla)):
            if x_i != x_j and x_i > x_j:
                qubo_matrix[x_i][x_j] = 0

    return qubo_matrix


def get_sol_sets(sets, sol_vec):
    res = []
    bin_vec = list(list(sol_vec)[0].values())[-len(sets):]
    for i in range(len(bin_vec)):
        if bin_vec[i] == 1:
            res.append(sets[i])
    return res


if __name__ == '__main__':
    sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage_system4.1'},
                                              token=''))

    U = [1, 2, 3, 4, 5]
    V = [[1, 2, 3], [2, 4], [3, 4], [4, 5]]

    Q = matrix_to_dict(set_cover_qubo(U, V))
    response = sampler.sample_qubo(Q, num_reads=100)
    sol = get_sol_sets(V, response.samples())
    print("The solution of the set cover problem U=" + str(U) + ", V=" + str(V) + " is: " + str(sol))
