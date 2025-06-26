import os
import argparse
import numpy as np
import scipy.sparse
from itertools import combinations
import random


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """

    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_SC(nrows, ncols, density, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i + n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i + n] = rng.choice(remaining_rows, size=i + n - nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr),
        shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([" +{} x{}".format(c[j], j + 1) for j in range(ncols)]))
        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([" +1 x{}".format(j + 1) for j in indices[indptr[i]:indptr[i + 1]]])
            file.write("C{}:".format(i) + row_cols_str + " >= 1\n")
        file.write("\nbinary\n")
        file.write("".join([" x{}".format(j + 1) for j in range(ncols)]))


def generate_CA(random, filename, n_items=100, n_bids=500, min_value=1, max_value=100,
                           value_deviation=0.5, n_item_per_bidder=5,
                           additivity=0.2):

    assert min_value >= 0 and max_value >= min_value

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)
    bids = []

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        items_chosen = random.choice(n_items, p=prob, replace=False, size=n_item_per_bidder)

        # compute bundle price with value additivity
        price = private_values[items_chosen].sum() + np.power(n_item_per_bidder, 1 + additivity)

        # place bids
        bids.append((list(items_chosen), price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items)]

        file.write("minimize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(" -{} x{}".format(price, i + 1))
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(" +1 x{}".format(i + 1))
                file.write(" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(" x{}".format(i + 1))


def generate_MIS(graph, filename, same_nedges=False, nedges=0):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """

    with open(filename, 'w') as lp_file:
        edges = graph.edges
        if same_nedges:
            edges = set(random.shuffle(list(edges))[:nedges])
        lp_file.write("minimize\nOBJ:" + "".join([" - 1 x{}".format(node + 1) for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, edge in enumerate(edges):
            lp_file.write(
                "C{}:".format(count + 1) + "".join([" + x{}".format(node + 1) for node in edge]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join(["x{}".format(node + 1) for node in range(len(graph))]) + "\n")


def generate_MVC(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """

    node_weight = np.random.random(len(graph))

    with open(filename, 'w') as lp_file:
        lp_file.write("minimize\nOBJ:" + "".join(
            [" + {} x{}".format(node_weight[node], node + 1) for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, edge in enumerate(graph.edges):
            lp_file.write(
                "C{}:".format(count + 1) + "".join([" + x{}".format(node + 1) for node in edge]) + " >= 1\n")
        lp_file.write("\nbinary\n" + " ".join(["x{}".format(node + 1) for node in range(len(graph))]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instance', type=str, default='MIS')
    parser.add_argument('--usage', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--number', type=int, default=1)
    args = parser.parse_args()
    seed = args.seed
    rng = np.random.RandomState(seed)
    random.seed(seed)
    lp_dir = f'./instance/{args.instance}/{args.usage}'

    if args.instance == 'SC':
        if args.usage == 'train':
            nrows = 5000
            ncols = 4000
            dens = 0.05
            max_coef = 100
        else:
            nrows = 20000
            ncols = 16000
            dens = 0.05
            max_coef = 100

        print("{} instances in {}".format(args.number, lp_dir))
        if not os.path.isdir(lp_dir):
            os.makedirs(lp_dir)

        for i in range(args.number):
            filename = f'{lp_dir}/instance_{i + 1}.lp'
            print('generating file {} ...'.format(filename))
            generate_SC(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

        print('done.')



    elif args.instance == 'CA':
        if args.usage == 'train':
            number_of_items = 2000
            number_of_bids = 4000
            number_of_items_per_bidder = 6
        else:
            number_of_items = 800000
            number_of_bids = 100000
            number_of_items_per_bidder = 40

        print("{} instances in {}".format(args.number, lp_dir))
        if not os.path.isdir(lp_dir):
            os.makedirs(lp_dir)

        # actually generate the instances
        for i in range(args.number):
            filename = f'{lp_dir}/instance_{i + 1}.lp'
            generate_CA(rng, filename, n_items=number_of_items, n_bids=number_of_bids, n_item_per_bidder=number_of_items_per_bidder)
            print(filename)
        print("done.")



    elif args.instance == 'MIS':
        if args.usage == 'train':
            num_nodes = 6000
            average_degree = 5
            edge_prob = average_degree / (num_nodes - 1)
        else:
            num_nodes = 100000
            average_degree = 100
            edge_prob = average_degree / (num_nodes - 1)

        print("{} instances in {}".format(args.number, lp_dir))
        if not os.path.isdir(lp_dir):
            os.makedirs(lp_dir)

        for i in range(args.number):
            filename = f'{lp_dir}/instance_{i + 1}.lp'
            generate_MIS(Graph.erdos_renyi(num_nodes, edge_prob, rng), filename, False)
            print(filename)


    elif args.instance == 'MVC':
        if args.usage == 'train':
            num_nodes = 1000
            affinity = 70
        else:
            num_nodes = 20000
            affinity = 200

        print("{} instances in {}".format(args.number, lp_dir))
        if not os.path.isdir(lp_dir):
            os.makedirs(lp_dir)

        for i in range(args.number):
            filename = f'{lp_dir}/instance_{i + 1}.lp'
            generate_MVC(Graph.barabasi_albert(num_nodes, affinity, rng), filename)
            print(filename)

