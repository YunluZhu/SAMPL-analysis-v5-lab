import random
import numpy as np
import pandas as pd

class Node:
    def __init__(self, env, node_id, neighbors, weights):
        self.env = env
        self.node_id = node_id
        self.neighbors = neighbors
        self.weights = weights
        self.current_neighbor = None

    # def start(self):
    #     self.current_neighbor = self.choose_target()
    #     while True:
    #         yield self.env.timeout(1)
    #         self.current_neighbor = self.choose_target()

    def choose_target(self):
        neighbors = list(self.neighbors)
        weights = list(self.weights)
        return random.choices(neighbors, weights=weights)[0]

    def run_simulation(self, start_node, sim_time):
        self.current_neighbor = start_node
        simfish_history = [(0, self.current_neighbor)]
        for i in range(sim_time):
            yield self.env.timeout(1)
            self.current_neighbor = self.choose_target()
            simfish_history.append((i+1, self.current_neighbor))
        return simfish_history

class Network:
    def __init__(self, env, edges, source_col, target_col, weight_col):
        source = source_col
        target = target_col
        weight = weight_col
        self.env = env
        self.nodes = {}
        nodes = set(edges[source]).union(set(edges[target]))
        for node_id in nodes:
            neighbors = list(edges[edges[source] == node_id][target])
            weights = list(edges[edges[source] == node_id][weight])
            node = Node(self.env, node_id, neighbors, weights)
            self.nodes[node_id] = node

    # def start(self):
    #     for node_id in self.nodes:
    #         self.env.process(self.nodes[node_id].start())
            
    def run_simfish(self, start_node, sim_time):
        simfish_history = []
        node = self.nodes[start_node]
        node.current_neighbor = start_node
        simfish_history.append(node.current_neighbor)
        for i in range(sim_time):
            self.env.run(until=self.env.now + 1)
            node.current_neighbor = node.choose_target()
            simfish_history.append(node.current_neighbor)
        return simfish_history

    def check_nodes(self):
        node_info = pd.DataFrame(data={
            'node_id': self.nodes.keys(),
            'neighbors': [node.neighbors for node in self.nodes.values()],
            'weights': [node.weights for node in self.nodes.values()]
        })
        return node_info