import networkx as nx
import random
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

def assign_topic_distributions(G, m):
    """
    Assigns a probability distribution over m topics for each node in the graph.

    Args:
        G (networkx.Graph): Input graph.
        m (int): Number of topics.

    Returns:
        dict: A dictionary where keys are nodes, and values are arrays representing
              the probability distribution over m topics.
    """
    topic_distributions = {}
    
    for node in G.nodes:
        # Generate a random probability distribution over m topics
        prob_distribution = np.random.dirichlet(alpha=np.ones(m), size=1)[0]
        topic_distributions[node] = prob_distribution
    
    return topic_distributions


# Load the Karate Club dataset
G = nx.karate_club_graph()

print(f'# of nodes = {len(G.nodes())}')
print(f'# of edges = {len(G.edges())}')

def sample_edges_and_components(G, p):
    """
    Samples edges from a graph with probability p and computes the connected components.

    Args:
        G (networkx.Graph): Input graph.
        p (float): Probability of sampling each edge.

    Returns:
        dict: A dictionary where keys are nodes and values are lists of nodes
              in the same connected component.
    """
    # Create a new graph with the same nodes as G
    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(G.nodes)
    
    # Sample edges with probability p
    for edge in G.edges:
        if random.random() < p:  # Keep the edge with probability p
            sampled_graph.add_edge(*edge)
    
    # Find connected components
    components = list(nx.connected_components(sampled_graph))
    
    # Build the dictionary mapping nodes to their components
    node_to_component = {}
    for component in components:
        for node in component:
            node_to_component[node] = set(component)
    
    return node_to_component, components

p = 0.1 # Sampling probability
print(f'sampling probability = {p}')
node_to_component, components = sample_edges_and_components(G, p)

m = 5  # Number of topics
print(f'# of topics = {m}')
topics = list(range(1, m + 1))
# Dictionary to store the topics assigned to each node
node_to_topics = {}

for node in G.nodes:
    # Randomly assign a subset of topics to the node
    assigned_topics = [topic for topic in topics if random.random() < 2/m]
    node_to_topics[node] = set(assigned_topics)

T = 100
wtps = []
for t in range(T):
    print(f't = {t}')
    node_to_component, components = sample_edges_and_components(G, p)
    random_topic = random.randint(1, m)
    potentials = []
    weights = []
    for node in G.nodes:
        #find which nodes cover the node and the random topic
        ws = np.array([1 if node in node_to_component[v] and random_topic in node_to_topics[v] else 0 for v in G.nodes()])
        potentials.append(Potential(b=1, w=ws))
        weights.append(1)
    
    # Keep potentials if weight > 0
    potentials = [potentials[i] for i in range(len(weights)) if weights[i] > 0]
    weights = [weight for weight in weights if weight > 0]

    wtps.append(WTP(potentials=potentials, weights=weights))

file = './instances/influence.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")

