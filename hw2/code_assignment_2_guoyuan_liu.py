import numpy as np
from tqdm import tqdm
from itertools import permutations


def generate_data(n, q, ns, ps, seed=0):
    """
    Generate data for Stochastic Block Model
    
    Input
    n::int: number of nodes
    q::int: number of groups
    ns::List[Float]: expected fraction of nodes per group
    ps::ndarray[(q, q), Float]: probability of edge between nodes

    Return
    g::ndarray[(n, ), int]: assignment of each node
    A::ndarray[(n, n), int]: adjacency matrix for directed graph G

    """
    g = np.zeros(n, dtype=int)
    A = np.zeros((n, n), dtype=int)
    rng = np.random.default_rng(seed=seed)

    for i in range(n):
        for a in range(q):
            if rng.uniform(0, 1, 1)[0] > ns[a]:
                g[i] = a
                break

    for i in range(n):
        for j in range(n):
            a = g[i]
            b = g[j]
            if rng.uniform(0, 1, 1)[0] < ps[a, b]:
                A[i, j] = 1
            else:
                A[i, j] = 0

    return g, A

def energy(g, A, ns, ps):
    """
    Compute the energy of the graph G

    Input
    g::ndarray[(n, ), int]: assignment of each node
    A::ndarray[(n, n), int]: adjacency matrix for directed graph G
    ns::List[Float]: expected fraction of nodes per group
    ps::ndarray[(q, q), float]: probability of edge between nodes

    Return
    E::Float: energy of the graph G

    """
    n = len(g)
    q = len(ns)
    E = 0
    for i in range(n):
        E -= np.log(ns[g[i]])
        for j in range(n):
            if i == j:
                continue
            a = g[i]
            b = g[j]
            E -= A[i, j] * np.log(ps[a, b]) + (1-A[i,j]) * np.log(1-ps[a,b])
    return E

def energy_difference(g1, g2, ns, ps, A):
    """
    Compute the energy difference between two
    node configurations.
    """
    return energy(g1, A, ns, ps) - energy(g2, A, ns, ps)

def overlap(g, g_star, ns):
    """
    Compute the overlap between a state g, and the ground truth, g_star. 
    The groups are indistinguishable (index of the group is meaningless),
    so we need to permute the group assignment.
    """
    n = len(g)
    q = len(ns)
    maxn = np.max(ns)
    overlap = -np.inf

    idx = []
    for i in range(q):
        idx.append(np.where(g_star == i)[0])

    for group_perm in permutations(range(q)):
        g_perm = np.zeros(n, dtype=int)
        for i in range(q):
            g_perm[idx[i]] = group_perm[i]
        
        tmp = (np.sum(g_perm == g_star) / n - maxn) / (1 - maxn)
        if overlap < tmp:
            overlap = tmp
    return overlap


def run_mcmc(A, ns, ps, n_steps, seed=0):
    """
    Run MCMC for Stochastic Block Model using Metropolis-Hastings algorithm.

    Input
    A::ndarray[(n, n), int]: adjacency matrix for directed graph G
    ns::List[Float]: expected fraction of nodes per group
    ps::ndarray[(q, q), float]: probability of edge between nodes among groups
    n_steps::Int: number of MCMC steps

    Return
    g_states::List[ndarray[(n, ], int]: history of nodes assignment
    """
    n = A.shape[0]
    q = len(ns)
    rng = np.random.default_rng(seed=seed)
    
    g_states = []
    # sample initial configuration
    g = np.zeros(n, dtype=int)
    for i in range(n):
        for a in range(q):
            if rng.uniform(0, 1, 1)[0] > ns[a]:
                g[i] = a
                break
    g_states.append(g.copy())

    for t in tqdm(range(n_steps)):
        i = rng.integers(0, n, 1)[0]
        g_new = g.copy()
        g_new[i] = 1 - g[i] # flip the assignment of node i, only works for q=2
        delta_E = energy_difference(g_new, g, ns, ps, A)
        if rng.uniform(0, 1, 1)[0] < np.min([1, np.exp(-delta_E)]):
            g = g_new
        g_states.append(g.copy())

    return g_states

def run_expectation_maximization(A, ns0, ps, n_steps, m_steps, seed=0):
    """
    Run Expectation Maximization for Stochastic Block Model.

    Input
    ns0::List[float]: initial guess for expected fraction of nodes per group

    Return
    ns_history::List[List[float]]: history of ns
    """
    q = len(ns0)
    n = A.shape[0]
    ns_history = []
    for m in range(m_steps):
        g_states = run_mcmc(A, ns0, ps, n_steps, seed=seed)
        ns_history.append(list(map(lambda i: np.sum(g_states[-1] == i) / n, range(q))))
        ns0 = ns_history[-1].copy()
    return ns_history


