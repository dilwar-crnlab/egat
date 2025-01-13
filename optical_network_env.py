import copy
import heapq
import random
from typing import List, Optional, Tuple

import gym
import networkx as nx
import numpy as np

from optical_rl_gym.utils import Service
from optical_rl_gym.utils import Service, Path, get_k_shortest_paths, get_path_weight




class OpticalNetworkEnv(gym.Env):
    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10.0,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 80,
        allow_rejection: bool = False,
        node_request_probabilities: Optional[np.array] = None,
        seed: Optional[int] = None,
        channel_width: float = 12.5,
        k_paths=5
    ):
        assert topology is None or "ksp" in topology.graph
        assert topology is None or "k_paths" in topology.graph
        self._events: List[Tuple[float, Service]] = []
        self.current_time: float = 0
        self.episode_length: int = episode_length
        self.services_processed: int = 0
        self.services_accepted: int = 0
        self.episode_services_processed: int = 0
        self.episode_services_accepted: int = 0

        self.current_service: Service = None
        self._new_service: bool = False
        self.allow_rejection: bool = allow_rejection

        self.load: float = 0
        self.mean_service_holding_time: float = 0
        self.mean_service_inter_arrival_time: float = 0
        self.set_load(load=load, mean_service_holding_time=mean_service_holding_time)

        self.rand_seed: Optional[int] = None
        self.rng: random.Random = None
        self.seed(seed=seed)


        # Convert topology to directed if it's not already xd950
        # if not isinstance(topology, nx.DiGraph):
        #     self.topology = nx.DiGraph(topology)
        # else:
        #     self.topology = copy.deepcopy(topology)
        self.topology: nx.Graph = copy.deepcopy(topology)
        #print("Topology edges:",self.topology.edges)

        # Edge indexing for directed edges
        # edge_index = 0
        # for u, v in self.topology.edges():
        #     self.topology[u][v]['index'] = edge_index
        #     edge_index += 1

        self.topology_name: str = topology.graph["name"]
        self.k_paths: int = self.topology.graph["k_paths"]
        print("ksp", self.k_paths)
        self.compute_node_degrees(self.topology)
        self.compute_edge_betweenness(self.topology)
        #self.print_edge_indices()

        # just as a more convenient way to access it
        self.k_shortest_paths = self.topology.graph["ksp"]
        assert (node_request_probabilities is None or len(node_request_probabilities) == self.topology.number_of_nodes())
        
        self.num_spectrum_resources: int = num_spectrum_resources

        # channel width in GHz
        self.channel_width: float = channel_width
        self.topology.graph["num_spectrum_resources"] = num_spectrum_resources

        # self.topology.graph["available_spectrum"] = np.full(
        #     (self.topology.number_of_edges()),
        #     fill_value=self.num_spectrum_resources,
        #     dtype=int
        # )
        
        # Initialize spectrum for directed edges
        self.topology.graph["available_spectrum"] = np.full(
            (self.topology.number_of_edges()),
            fill_value=self.num_spectrum_resources,
            dtype=int
        )

        if node_request_probabilities is not None:
            self.node_request_probabilities = node_request_probabilities
        else:
            self.node_request_probabilities = np.full((self.topology.number_of_nodes()),fill_value=1.0 / self.topology.number_of_nodes(),)
            
    def compute_node_degrees(self, topology):
        """
        Computes the degree of each node and stores it in topology.graph.
        Also computes and stores the average, minimum, and maximum node degrees.
        
        The degree information is stored in:
        - topology.graph['node_degrees']: Dictionary of node degrees
        - topology.graph['avg_node_degree']: Average node degree
        - topology.graph['min_node_degree']: Minimum node degree
        - topology.graph['max_node_degree']: Maximum node degree
        """
        # Initialize dictionary to store node degrees
        node_degrees = {}
        
        # Compute degree for each node
        for node in self.topology.nodes():
            degree = self.topology.degree(node)
            node_degrees[node] = degree
        
        # Calculate statistics
        degrees = list(node_degrees.values())
        avg_degree = sum(degrees) / len(degrees)
        min_degree = min(degrees)
        max_degree = max(degrees)
        
        # Store in topology.graph
        self.topology.graph['node_degrees'] = node_degrees
        self.topology.graph['avg_node_degree'] = avg_degree
        self.topology.graph['min_node_degree'] = min_degree
        self.topology.graph['max_node_degree'] = max_degree


    # Method to print edge indices
    def print_edge_indices(self):
        print("\nEdge Indices:")
        print("Edge (source->destination) : Index")
        print("---------------------------------")
        for u, v in self.topology.edges():
            edge_index = self.topology[u][v]['index']
            print(f"Edge ({u}->{v}) : {edge_index}")

    def compute_edge_betweenness(self, topology):
        # Compute edge betweenness centrality
        edge_betweenness = nx.edge_betweenness_centrality(self.topology)
        
        # Find max for normalization
        max_betweenness = max(edge_betweenness.values())
        
        # Store normalized values in topology
        for edge in self.topology.edges():
            node1, node2 = edge
            normalized_betweenness = edge_betweenness[edge] / max_betweenness
            self.topology[node1][node2]['betweenness'] = normalized_betweenness

        # Also store max value for reference
        self.topology.graph['max_betweenness'] = max_betweenness

    def set_load(self, load: float = None, mean_service_holding_time: float = None) -> None:
        """
        Sets the load to be used to generate requests.
        :param load: The load to be generated, in Erlangs
        :param mean_service_holding_time: The mean service holding time to be used to
        generate the requests
        :return: None
        """
        if load is not None:
            self.load = load
        if mean_service_holding_time is not None:
            self.mean_service_holding_time = (mean_service_holding_time ) # current_service holding time in seconds
            
        self.mean_service_inter_arrival_time = 1 / float(self.load / float(self.mean_service_holding_time))
        

    def _plot_topology_graph(self, ax) -> None:
        pos = nx.get_node_attributes(self.topology, "pos")
        nx.draw_networkx_edges(self.topology, pos, ax=ax)
        nx.draw_networkx_nodes(
            self.topology,
            pos,
            nodelist=[
                x
                for x in self.topology.nodes()
                if x in [self.current_service.source, self.current_service.destination]
            ],
            label=[x for x in self.topology.nodes()],
            node_shape="s",
            node_color="white",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.topology,
            pos,
            nodelist=[
                x
                for x in self.topology.nodes()
                if x
                not in [self.current_service.source, self.current_service.destination]
            ],
            label=[x for x in self.topology.nodes()],
            node_shape="o",
            node_color="white",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_labels(self.topology, pos)
        nx.draw_networkx_edge_labels(
            self.topology,
            pos,
            edge_labels={
                (i, j): "{}".format(
                    self.available_spectrum[self.topology[i][j]["index"]]
                )
                for i, j in self.topology.edges()
            },
        )
        # TODO: implement a trigger (a flag) that tells whether to plot the edge labels
        # set also an universal label dictionary inside the edge dictionary, e.g.,
        # (self.topology[a][b]['plot_label']

    def _add_release(self, service: Service) -> None:
        """
        Adds an event to the event list of the simulator.
        This implementation is based on the functionalities of heapq:
        https://docs.python.org/2/library/heapq.html

        :param event:
        :return: None
        """
        heapq.heappush(
            self._events, (service.arrival_time + service.holding_time, service)
        )

    def _get_node_pair(self) -> Tuple[str, int, str, int]:
        """
        Uses the `node_request_probabilities` variable to generate a source and a destination.

        :return: source node, source node id, destination node, destination node id
        """
        src = self.rng.choices([x for x in self.topology.nodes()], weights=self.node_request_probabilities)[0]
        src_id = self.topology.graph["node_indices"].index(src)
        new_node_probabilities = np.copy(self.node_request_probabilities)
        new_node_probabilities[src_id] = 0.0
        new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
        dst = self.rng.choices([x for x in self.topology.nodes()], weights=new_node_probabilities)[0]
        dst_id = self.topology.graph["node_indices"].index(dst)
        return src, src_id, dst, dst_id

    def observation(self):
        return {"topology": self.topology, "service": self.current_service}

    def reward(self):
        return 1 if self.current_service.accepted else -1

    def reset(self) -> None:
        self._events = []
        self.current_time = 0
        self.services_processed = 0
        self.services_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.topology.graph["available_spectrum"] = np.full(
            self.topology.number_of_edges(), 
            fill_value=self.num_spectrum_resources,
            dtype=int,
        )
        # # Initialize spectrum resources for directed edges
        # self.topology.graph["available_slots"] = np.ones(
        #     (self.topology.number_of_edges(), self.num_spectrum_resources), 
        #     dtype=int
        # )

        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []

        self.topology.graph["last_update"] = 0.0
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["utilization"] = 0.0
            self.topology[lnk[0]][lnk[1]]["last_update"] = 0.0
            self.topology[lnk[0]][lnk[1]]["services"] = []
            self.topology[lnk[0]][lnk[1]]["running_services"] = []

    def seed(self, seed=None):
        if seed is not None:
            self.rand_seed = seed
        else:
            self.rand_seed = 41
        self.rng = random.Random(self.rand_seed)
