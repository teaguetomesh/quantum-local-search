import networkx as nx

def boppana_halldorsson(G, iset=None):
    r"""Adapted from the NetworkX implementation to support initial independent sets.

    Repeatedly remove cliques from the graph.
    Results in a $O(|V|/(\log |V|)^2)$ approximation of maximum clique
    and independent set. Returns the largest independent set found, along
    with found maximal cliques.
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph
    Returns
    -------
    max_ind_cliques : (set, list) tuple
        2-tuple of Maximal Independent Set and list of maximal cliques (sets).
    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    if iset:
        # We have been given an intial independent set, only consider the subgraph
        # induced by V / {n, N(n): n in iset} where N(n) is the neighbors of node n.
        invalid_nodes = iset.copy()
        for n in iset:
            invalid_nodes.extend([nbr for nbr in nx.all_neighbors(G, n) if nbr != n])
        invalid_nodes = list(set(invalid_nodes))
        graph = G.copy().subgraph([v for v in G.nodes if v not in invalid_nodes]).copy()
    else:
        graph = G.copy()
    c_i, i_i = nx.algorithms.approximation.ramsey.ramsey_R2(graph)
    cliques = [c_i]
    isets = [i_i]
    while graph:
        graph.remove_nodes_from(c_i)
        c_i, i_i = nx.algorithms.approximation.ramsey.ramsey_R2(graph)
        if c_i:
            cliques.append(c_i)
        if i_i:
            isets.append(i_i)
    # Determine the largest independent set as measured by cardinality.
    maxiset = max(isets, key=len)
    return maxiset
