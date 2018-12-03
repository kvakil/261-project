import networkx as nx
import random
import heapq
from math import log1p

# Number of out-going connections allowed per client. This bound is only met in
# expectation, but it shouldn't affect our simulation too much
OUTGOING = 8

# Number of nodes in the bitcoin network
NODES = 256

# Nodes we are "probing" by sending a transaction to.
PROBE_SET = set(range(NODES / 2))

# Use our modification to use the double spend
USE_DOUBLE_SPEND = False

# Tags for networkx graph
ADVERSARY = 'adv'
ORIGINATOR = 'ori'

def _annotate_direction(G):
    for edge in G.edges():
        which_started = random.choice(edge)
        nx.set_edge_attributes(G, name=ORIGINATOR, values={edge: which_started})

def _generate_bitcoin_graph():
    """Generate a bitcoin graph where each node has approximately OUTGOING
    outgoing connections and OUTGOING incoming connections."""
    G = nx.generators.random_graphs.random_regular_graph(2 * OUTGOING, NODES)
    _annotate_direction(G)
    return G

def _connect_adversary(G):
    """Make the adversary create an outgoing connection to each node on the graph"""
    G.add_node(ADVERSARY)
    for node in G.nodes():
        if node != ADVERSARY:
            edge = (node, ADVERSARY)
            G.add_edge(*edge)
            nx.set_edge_attributes(G, name=ORIGINATOR, values={edge: ADVERSARY})

def delta_next_time_to_send(G, u, v):
    """How long to wait before U should send a message to V under diffusion
    spreading. Per the Bitcoin protocol, this depends on if we have an outgoing
    connection or an incoming connection."""
    is_outgoing = G[u][v][ORIGINATOR] == u
    average_interval_seconds = 2 if is_outgoing else 5
    delta = int(log1p(-random.random()) * average_interval_seconds * -1000000 + 0.5)
    return delta if delta > 0 else 0

def generate_adversary_graph():
    """A graph with the adversary connection to a fraction SPY_P of the nodes"""
    G = _generate_bitcoin_graph()
    _connect_adversary(G)
    G = nx.freeze(G)
    return G

def undirected(edge):
    """Given EDGE, return an canonicalized undirected version."""
    return tuple(sorted(edge))

# tag for priority queue simulation
SEND_TRANSACTIONS = 'ST'

class Transaction(object):
    """A transaction."""
    def __init__(self, utxos, meta):
        self.utxos = set(utxos)
        self.meta = meta

    def __eq__(self, other):
        # If our UXTOs intersect, then only one should be allowed, as otherwise
        # a double spend occurs
        return bool(self.utxos & other.utxos)

    def __hash__(self):
        # If this is a double spend attack, we need to ensure all transactions
        # fall into the same hashmap bucket
        if USE_DOUBLE_SPEND:
            return 0
        else:
            return hash(tuple(self.utxos))

    def __str__(self):
        return str((self.utxos, self.meta))

    def __repr__(self):
        return str((self.utxos, self.meta))

class DiffusionSimulation(object):
    def __init__(self, topology):
        self.G = topology

        # Priority queue for simulation
        self.pq = []

        # A mapping from DIRECTED edge => queue of transactions which will be
        # sent on that edge
        self.tx_queue = dict()

        # A mapping from UNDIRECTED edge => transactions which both
        # participants know
        self.tx_known = dict()

        for edge in self.G.edges():
            self.tx_known[undirected(edge)] = set()

        # A mapping from nodes => mempool transactions
        self.tx_mempool = dict()

        for node in self.G.nodes():
            self.tx_mempool[node] = set()

        # Guesses based on first timestamp
        self.first_timestamp = dict()

        # Current time in simulation
        self.current_time = 0

        # We save a log factor by heapifying at the end here
        for edge in self.G.edges():
            self.pq.append(self.get_next_broadcast_time(edge))
            self.pq.append(self.get_next_broadcast_time(edge[::-1]))
        heapq.heapify(self.pq)

        # A map from nodes => connections we've guessed based on first timestamp
        self.guessed_connections = dict()

    def get_next_broadcast_time(self, edge):
        """Get the next (absolute) time we will need to send transactions along
        the given edge, suitable for insertion into the priority queue."""
        return ((self.current_time + delta_next_time_to_send(G, *edge), edge, SEND_TRANSACTIONS))

    def adversary_broadcast(self):
        for node in self.G[ADVERSARY]:
            if node in PROBE_SET:
                edge = (ADVERSARY, node)
                if USE_DOUBLE_SPEND:
                    # All transactions attempt to spend the UTXO -1, and also
                    # attempt to spend a different UTXO (per node).
                    tx = set([Transaction((-1, node), node)])
                else:
                    tx = set([Transaction((node, ), node)])
                self.tx_mempool[node] |= tx
                self.tx_known[undirected(edge)] |= tx

    def step(self):
        """Step in the simulation"""
        if not self.pq:
            return False
        new_time, edge, action = heapq.heappop(self.pq)
        if action == SEND_TRANSACTIONS:
            src, dst = edge
            if src != ADVERSARY: # adversary does not broadcast after start
                transactions = self.tx_mempool[src]
                # Remove transactions they already know about
                to_send = transactions - self.tx_known[undirected(edge)]

                # First timestamp reporting
                if dst == ADVERSARY and src not in PROBE_SET:
                    if to_send:
                        if src not in self.first_timestamp:
                            self.first_timestamp[src] = (new_time, max(random.choice(list(to_send)).utxos))
                        if len(self.first_timestamp) == NODES - len(PROBE_SET): # are we done?
                            return False

                # Transactions in to_send will be ignored if they conflict
                # with a known transaction, because they will compare equal
                # by our definition of __eq__.
                self.tx_known[undirected(edge)] |= to_send
                self.tx_mempool[dst] |= to_send

                heapq.heappush(self.pq, self.get_next_broadcast_time(edge))
        else:
            assert False, 'unknown requested action'

        self.current_time = new_time
        return True

def score(G, results):
    """The score is the number of edges we got correctly, divided by the number of edges we guessed."""
    points = 0
    for src, (_timestamp, detected_node) in results.items():
        if detected_node in G[src]:
            points += 1
    return points / float(len(results))

def truncate_after(d, n):
    """Truncate first timestamp dictionary D after N entries."""
    sorted_lst = sorted(d.items(), key=lambda a: a[1][0])
    return dict(sorted_lst[:n])

for _ in range(NODES):
    G = generate_adversary_graph()
    d = DiffusionSimulation(G)
    d.adversary_broadcast()
    while True:
        still_going = d.step()
        if not still_going or d.current_time >= 50000000:
            print(d.current_time, score(G, truncate_after(d.first_timestamp, 14)))
            break
