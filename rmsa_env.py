import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import gym
import networkx as nx
import numpy as np
import random
from optical_rl_gym.utils import Path, Service
from optical_rl_gym.osnr_calculator import *

from .optical_network_env import OpticalNetworkEnv




import logging

# Set global logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG, INFO, WARNING, etc.
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)


class RMSAEnv(OpticalNetworkEnv):

    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
    }

    def __init__(
        self,
        num_bands=None,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10,
        mean_service_holding_time: float = 10.0,
        #num_spectrum_resources: int = 100,
        #bit_rate_selection: str = "discrete",
        bit_rates: Sequence = [10, 40, 100],
        bit_rate_probabilities: Optional[np.array] = None,
        node_request_probabilities: Optional[np.array] = None,
        #bit_rate_lower_bound: float = 25.0,
        #bit_rate_higher_bound: float = 100.0,
        seed: Optional[int] = None,
        allow_rejection: bool = False,
        reset: bool = True,
        channel_width: float = 12.5,
        k_paths=5
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            #num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            channel_width=channel_width,
            k_paths=k_paths
        )

        # make sure that modulations are set in the topology
        #assert "modulations" in self.topology.graph

    
        self.physical_params = PhysicalParameters() # for using PhysicalParameters data class
        # Initialize OSNR calculator
        self.osnr_calculator = OSNRCalculator()
        self.num_bands = num_bands
        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["c_band_active_conns"] = []
            self.topology[lnk[0]][lnk[1]]["l_band_active_conns"] = []
            self.topology[lnk[0]][lnk[1]]['l_band_fragmentation'] = 0.0
            self.topology[lnk[0]][lnk[1]]['c_band_fragmentation'] = 0.0

        


        multi_band_spectrum_resources = [100, 256] #332 -> 100, 916->256
        if self.num_bands == 1:
            self.num_spectrum_resources = multi_band_spectrum_resources[0]
        elif self.num_bands == 2:
            self.num_spectrum_resources = multi_band_spectrum_resources[1]

        self.C_band_start = 0
        self.C_band_end = 99
        self.L_band_start = 100
        self.L_band_end = 255

        #bit error rate (BER) of 10−3 are 9 dB, 12dB, 16 dB, and 18.6 dB,
        self.OSNR_th ={
            'BPSK': 9,
            'QPSK': 12,
            '8QAM': 16,
            '16QAM': 18.6
        }
        # Frequency ranges for C and L bands (in THz)
        self.band_frequencies = {
            0: {  # C-band
                'start': 191.3e12,  # Hz
                'end': 196.08e12,    # THz
            },
            1: {  # L-band
                'start': 184.4e12,  # THz
                'end': 191.3e12,    # THz
            }
        }



        self.spectrum_usage = np.zeros((self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources), dtype=int)

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1, dtype=int)
        

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1, 
                                        self.num_bands + 1,
                                        self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_output = np.zeros((self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int)
        
        self.actions_taken = np.zeros((self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int)
        
        self.episode_actions_taken = np.zeros((self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int)
        
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                "topology": gym.spaces.Discrete(10),
                "current_service": gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )

        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)



    
    
    def compute_node_features(self, service):
        """
        Compute path-wise node features for nodes in candidate paths.
        Only includes betweenness centrality as the node feature.
        """
        path_wise_node_features = {}
        
        # Get paths for current service
        paths = self.k_shortest_paths[service.source, service.destination]
        
        # Compute betweenness centrality once for all nodes
        betweenness = nx.betweenness_centrality(self.topology)
        max_betweenness = max(betweenness.values())
        
        # Process each path
        for path_idx, path in enumerate(paths):
            # Get nodes in this path
            current_path_nodes = path.node_list
            # Store node features for this path
            path_node_features = {}
            for node in current_path_nodes:
                # Only compute normalized betweenness
                path_node_features[node] = {
                    'betweenness': betweenness[node] / max_betweenness if max_betweenness > 0 else 0
                }
            # Store features for this path
            path_wise_node_features[path_idx] = {
                'nodes': set(current_path_nodes),
                'node_features': path_node_features,
                'path_length': len(current_path_nodes)
            }
        return path_wise_node_features

    
    def compute_path_edge_features(self, service):
        """
        Computes path-wise edge features for candidate paths.
        """
        path_wise_features = {}  
        # Get paths for current service
        paths = self.k_shortest_paths[service.source, service.destination]
        max_path_length = self.get_max_path_length()
        def compute_band_features(slots, band_type, node1, node2):
            """
            Compute features for specific band (C or L)
            """
            total_slots = len(slots)
            active_conns = 'c_band_active_conns' if band_type == 0 else 'l_band_active_conns'
            band_start = 0 if band_type == 0 else 100
            # Get continuous blocks using RLE
            initial_indices, values, lengths = RMSAEnv.rle(slots)
            available_indices = np.where(values == 1)[0]
            # Initialize features array
            features = np.zeros(8)
            # 1. Spectrum Utilization
            features[0] = np.sum(slots) / total_slots
            # 2-3. First-fit block info
            if len(available_indices) > 0:
                first_idx = available_indices[0]
                features[1] = (initial_indices[first_idx] + band_start) / total_slots  # position
                features[2] = lengths[first_idx] / total_slots  # size
            else:
                features[1] = -1  # No available first block
                features[2] = 0   # No size
            
            # 4-5. Last-fit block info
            if len(available_indices) > 0:
                last_idx = available_indices[-1]
                features[3] = (initial_indices[last_idx] + band_start) / total_slots  # position
                features[4] = lengths[last_idx] / total_slots  # size
            else:
                features[3] = -1  # No available last block
                features[4] = 0   # No size
            # 6. Fragmentation
            features[5] = self._compute_shannon_entropy(slots)
            # 7. Traffic Load (number of active connections)
            active_connections = self.topology[node1][node2].get(active_conns, [])
            features[6] = len(active_connections) / self.episode_length
            # 8. Average OSNR margin
            features[7] = self.compute_avg_osnr_margin(node1, node2, active_conns)
            return features
        # Compute features path by path
        for path_idx, path in enumerate(paths):
            current_path_nodes = set()  # nodes in this path
            current_path_edges = []     # edges in this path
            edge_features = []          # features of edges in this path
            # Get edges and nodes for this path
            for i in range(len(path.node_list) - 1):
                node1, node2 = path.node_list[i], path.node_list[i + 1]
                current_path_edges.append((node1, node2))
                current_path_nodes.add(node1)
                current_path_nodes.add(node2)
                # Compute edge features
                link_idx = self.topology[node1][node2]['index']
                # C-band features
                c_band_slots = self.topology.graph['available_slots'][link_idx, :100]
                c_features = compute_band_features(c_band_slots, 0, node1, node2)
                # L-band features
                l_band_slots = self.topology.graph['available_slots'][link_idx + self.topology.number_of_edges(), 100:256]
                l_features = compute_band_features(l_band_slots, 1, node1, node2)
                # Combine C and L band features
                edge_feat = np.concatenate([c_features, l_features])
                edge_features.append(edge_feat)
            # Pad edge features to max_path_length
            while len(edge_features) < max_path_length:
                zero_features = np.zeros(16)  # 8 C-band + 8 L-band features
                edge_features.append(zero_features)
            # Store all information for this path
            path_wise_features[path_idx] = {
                'nodes': current_path_nodes,
                'edges': current_path_edges,
                'edge_features': np.array(edge_features),  # Shape: [max_path_length, 16]
                'path_length': len(path.node_list) - 1
            }
        return path_wise_features
        
    
    
    def compute_avg_osnr_margin(self, node1, node2, active_conns):
        """Compute average OSNR margin of active services."""
        osnr_margins = []
        for service in self.topology[node1][node2][active_conns]:
            osnr_margins.append(service.OSNR_margin) 
        avg_margin = np.mean(osnr_margins) if osnr_margins else 0
        return avg_margin
    

    def compute_adjacency_matrix(self):
        num_nodes = self.topology.number_of_nodes()
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for edge in self.topology.edges():
            node1, node2 = edge
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1  # Undirected graph
        return adjacency_matrix



    def _compute_shannon_entropy(self, slots):
        """
        Compute Shannon entropy-based fragmentation metric using RLE method.
        """
        if np.all(slots == 0) or np.all(slots == 1):
            return 0.0
            
        # Use RLE method to get blocks
        initial_indices, values, lengths = RMSAEnv.rle(slots)
        
        # Get indices of free blocks (where value is 1)
        free_indices = np.where(values == 1)[0]
        if len(free_indices) == 0:
            return 1.0  # Maximum fragmentation
        
        # Get lengths of free blocks
        free_lengths = lengths[free_indices]
        total_free = np.sum(free_lengths)
        
        # Calculate probabilities
        probabilities = free_lengths / total_free
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(free_indices))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_center_frequency(self, service: Service):
        # Get band frequency range
        service_end_idx= service.initial_slot + service.number_slots -1
        service_center = (service.initial_slot + service_end_idx)/2
        if service.band == 0:
            center_freq = self.band_frequencies[service.band]['start'] + (service_center * 12.5e9)
        elif service.band == 1:
            center_freq = self.band_frequencies[service.band]['start'] + (service_center - 100 ) * 12.5e9
        return center_freq

    def step(self, action: [int]):
        path, band, initial_slot = action[0], action[1], action[2]

        # registering overall statistics
        self.actions_output[path, band, initial_slot] += 1
        previous_network_compactness = (self._get_network_compactness())  

        # starting the service as rejected
        self.current_service.accepted = False
        if (path < self.k_paths and band < self.num_bands and initial_slot < self.num_spectrum_resources):  # action is for assigning a path
            temp_path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path]
            if temp_path.length <= 4000:
                #print("Temp path len", temp_path.length )

                slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], self.num_bands, band, self.modulations)
                self.logger.debug(
                    "{} processing action {} path {} and initial slot {} for {} slots".format(
                        self.current_service.service_id, action, path, initial_slot, slots))
                if self.is_path_free(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], initial_slot, slots, band ):
                        
                        #check for OSNR
                        temp_service = copy.deepcopy(self.current_service)
                        temp_service.bandwidth = slots * 12.5e9 # in GHz
                        temp_service.band = band
                        temp_service.initial_slot = initial_slot
                        temp_service.number_slots = slots
                        temp_service.path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path]

                        temp_service.center_frequency = self._calculate_center_frequency(temp_service)

                        temp_service.modulation_format = self.get_modulation_format(temp_path, self.num_bands, band, self.modulations)['modulation']
                        
                        #print("Temp serive:", temp_service)
                        osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                        #print("OSNR", osnr_db)
                        if osnr_db >= self.OSNR_th[temp_service.modulation_format]:
                            #print("OSNR", osnr)
                            self.current_service.current_OSNR = osnr_db
                            self.current_service.OSNR_th = self.OSNR_th[temp_service.modulation_format]
                            self.current_service.OSNR_margin = osnr_db - self.OSNR_th[temp_service.modulation_format]
                            # if so, provision it (write zeros the position os the selected block in the available slots matrix
                            self._provision_path(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path],
                                                initial_slot, slots, band, self.current_service.arrival_time)
                            self.current_service.accepted = True  # the request was accepted

                            self.actions_taken[path, band, initial_slot] += 1
                            self._add_release(self.current_service)
                else:
                    self.current_service.accepted = False  # the request was rejected (blocked), the path is not free
        else:
            self.current_service.accepted = False # the request was rejected (blocked), the path is not free
                

        if not self.current_service.accepted:
            self.actions_taken[self.k_paths, self.num_bands, self.num_spectrum_resources] += 1

        self.topology.graph["services"].append(self.current_service)

        # generating statistics for the episode info
        

        cur_network_compactness = (self._get_network_compactness())  # measuring compactness after the provisioning
        k_paths = self.k_shortest_paths[self.current_service.source, self.current_service.destination]
        path_selected = k_paths[path] if path < self.k_paths else None
        reward = self.reward(path_selected, band)
        info = {
            "band": band if self.services_accepted else -1,
            "service_blocking_rate": (self.services_processed - self.services_accepted)/ self.services_processed,
            "episode_service_blocking_rate": (self.episode_services_processed - self.episode_services_accepted)/ self.episode_services_processed,
            "bit_rate_blocking_rate": (self.bit_rate_requested - self.bit_rate_provisioned)/ self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned)/ self.episode_bit_rate_requested,
            "network_compactness": cur_network_compactness,
            "network_compactness_difference": previous_network_compactness- cur_network_compactness,
            "avg_link_compactness": np.mean([self.topology[lnk[0]][lnk[1]]["compactness"] for lnk in self.topology.edges()]),
            "avg_link_utilization": np.mean([self.topology[lnk[0]][lnk[1]]["utilization"] for lnk in self.topology.edges()]),
        }

        # informing the blocking rate per bit rate
        # sorting by the bit rate to match the previous computation
       

        self._new_service = False
        self._next_service()
        return (self.observation(), reward, self.episode_services_processed == self.episode_length, info)
        
    #def reward(self, band, path_selected):
    #    return super().reward()
    
    def reward(self, path, band):
        """
        Compute reward considering:
        1. Service acceptance
        2. OSNR margin of allocation
        3. Fragmentation impact
        4. Resource utilization
        """
        if not self.current_service.accepted:
            return -1
            
        # Get selected path
        #print(path)
        #path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][1]
        #print(path)
        reward = 0
        
        # 1. Base reward for acceptance
        reward += 1
        
        # 2. OSNR margin reward
        osnr_margin = self.current_service.OSNR_margin
        osnr_th = self.current_service.OSNR_th
        normalized_osnr = osnr_margin / osnr_th
        reward += 0.3 * normalized_osnr
        
        # 3. Fragmentation penalty
        total_fragmentation = 0
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            if band == 0:  # C-band
                frag = self.topology[node1][node2]['c_band_fragmentation']
            else:  # L-band
                frag = self.topology[node1][node2]['l_band_fragmentation']
            total_fragmentation += frag
        
        avg_fragmentation = total_fragmentation / (len(path.node_list) - 1)
        reward -= 0.2 * avg_fragmentation  # penalty for fragmentation
        
        # 4. Utilization balance reward
        total_utilization = 0
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            if band == 0:
                util = self.topology[node1][node2]['c_band_util']
            else:
                util = self.topology[node1][node2]['l_band_util']
            total_utilization += util
        
        avg_utilization = total_utilization / (len(path.node_list) - 1)
        # Reward for balanced utilization (closest to 0.5)
        balance_metric = 1 - 2 * abs(0.5 - avg_utilization)
        #reward += 0.2 * balance_metric

        # Band preference based on modulation
        modulation = self.current_service.modulation_format
        if modulation in ['BPSK', 'QPSK']:
            if band == 1:  # L-band
                reward += 0.2  # Bonus for using preferred band
            else:  # C-band
                reward -= 0.1  # Penalty for using non-preferred band
        elif modulation in ['8QAM', '16QAM']:
            if band == 0:  # C-band
                reward += 0.2  # Bonus for using preferred band
            else:  # L-band
                reward -= 0.1  # Penalty for using non-preferred band
        
        return reward
    
    def reset(self, only_episode_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )

        

        if only_episode_counters:
            if self._new_service:
                # initializing episode counters
                # note that when the environment is reset, the current service remains the same and should be accounted for
                self.episode_services_processed += 1
                self.episode_bit_rate_requested += self.current_service.bit_rate
                
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources), dtype=int
        )

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1, dtype=int)
        

        

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0
        for lnk in self.topology.edges():
            self.topology[lnk[0]][lnk[1]]["external_fragmentation"] = 0.0
            self.topology[lnk[0]][lnk[1]]["compactness"] = 0.0
            self.topology[lnk[0]][lnk[1]]["c_band_active_conns"] = []
            self.topology[lnk[0]][lnk[1]]["l_band_active_conns"] = []
            self.topology[lnk[0]][lnk[1]]['l_band_fragmentation'] = 0.0
            self.topology[lnk[0]][lnk[1]]['c_band_fragmentation'] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode="human"):
        return

    def _provision_path(self, path: Path, initial_slot, number_slots, band, at):
        # usage
        if not self.is_path_free(path, initial_slot, number_slots, band):
            raise ValueError("Path {} has not enough capacity on slots {}-{}".format(path.node_list, path, initial_slot, initial_slot + number_slots))


        self.logger.debug(
            "{} assigning path {} on initial slot {} for {} slots".format(self.current_service.service_id, path.node_list, initial_slot, number_slots))
        # computing horizontal shift in the available slot matrix
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        active_conns = 'c_band_active_conns' if self.current_service.band == 0 else 'l_band_active_conns'
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]

            self.topology.graph["available_slots"][link_idx + (self.topology.number_of_edges() * band), initial_slot_shift : initial_slot_shift + number_slots] = 0
            
            self.spectrum_slots_allocation[
                link_idx + (self.topology.number_of_edges() * band), initial_slot_shift : initial_slot_shift + number_slots] = self.current_service.service_id

            self.topology[node1][node2]["services"].append(self.current_service)
            self.topology[node1][node2]["running_services"].append(self.current_service) #can be used for finding number of active connections

            self.topology[node1][node2][active_conns].append(self.current_service) #added by xd950
            

            #self._update_link_stats(path.node_list[i], path.node_list[i + 1])
            self._update_band_stats(node1, node2, band)


        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.path = path
        self.current_service.band = band
        self.current_service.initial_slot = initial_slot_shift
        self.current_service.number_slots = number_slots
        self.current_service.bandwidth = number_slots * 12.5e9
        #self.service.modulation_format = 

        self.current_service.termination_time = self.current_time + at
        self.current_service.center_frequency = self._calculate_center_frequency(self.current_service) #Error with the band
        #self.service.accepted = True  # the request was accepted
        self._update_network_stats()

        self.update_Service_OSNR(path, band) #only update the OSNR of existing service in [node1][node2][active_conn]


        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate

    def update_Service_OSNR(self, path, band):

        active_conns = 'c_band_active_conns' if band == 0 else 'l_band_active_conns'
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i + 1]

            for service in self.topology[node1][node2][active_conns]:
                
                #print("")
                updated_OSNR = self.osnr_calculator.calculate_osnr(service, self.topology)
                #print("Updated OSNR", updated_OSNR)
                #print(node1, "-->", node2, service.service_id, service.current_OSNR,"Updated OSNR", updated_OSNR, "Chnage in OSNR", updated_OSNR-service.current_OSNR)
            #print(self.topology[node1][node2][active_conns])


    def _release_path(self, service: Service):
        # Get active connections list based on band
        active_conns = 'c_band_active_conns' if self.current_service.band == 0 else 'l_band_active_conns'

        for i in range(len(service.path.node_list) - 1):
            node1, node2 = service.path.node_list[i], service.path.node_list[i + 1]
            link_idx = self.topology[node1][node2]["index"]
    
            # Free spectrum slots

            self.topology.graph["available_slots"][
                link_idx + (self.topology.number_of_edges() * service.band),
                service.initial_slot : service.initial_slot + service.number_slots
            ] = 1

            # Clear spectrum allocation
            self.spectrum_slots_allocation[
                link_idx + (self.topology.number_of_edges() * service.band),
                service.initial_slot : service.initial_slot + service.number_slots,
            ] = -1

            self.topology[node1][node2]["running_services"].remove(service)
            self.topology[node1][node2][active_conns].remove(service)
            self._update_band_stats(node1, node2, service.band )
            
        self.topology.graph["running_services"].remove(service)
        # Remove service from band-specific tracking
    

    def _update_network_stats(self):
        last_update = self.topology.graph["last_update"]
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph["throughput"]
            last_compactness = self.topology.graph["compactness"]

            cur_throughput = 0.0

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = ((last_throughput * last_update) + (cur_throughput * time_diff)) / self.current_time
            self.topology.graph["throughput"] = throughput

            compactness = ((last_compactness * last_update) + (self._get_network_compactness() * time_diff)) / self.current_time
            self.topology.graph["compactness"] = compactness

        self.topology.graph["last_update"] = self.current_time


    def _update_band_stats(self, node1: str, node2: str, band: int):
        """
        Updates metrics for specified band using time-weighted averaging.
        Args:
            node1, node2: Link endpoints
            band: 0 for C-band, 1 for L-band
        """
        last_update = self.topology[node1][node2]["last_update"]
        time_diff = self.current_time - last_update
        link_idx = self.topology[node1][node2]["index"]

        if self.current_time > 0:
            # Get band-specific parameters
            if band == 0:  # C-band
                band_slots = self.topology.graph["available_slots"][link_idx, :100]
                total_slots = 100
                last_util = self.topology[node1][node2].get("c_band_util", 0.0)
                last_frag = self.topology[node1][node2].get("c_band_fragmentation", 0.0)
                util_key = "c_band_util"
                frag_key = "c_band_fragmentation"
            else:  # L-band
                band_slots = self.topology.graph["available_slots"][
                    link_idx + self.topology.number_of_edges(), 100:256]
                total_slots = 156
                last_util = self.topology[node1][node2].get("l_band_util", 0.0)
                last_frag = self.topology[node1][node2].get("l_band_fragmentation", 0.0)
                util_key = "l_band_util"
                frag_key = "l_band_fragmentation"

            # Calculate current metrics
            cur_util = (total_slots - np.sum(band_slots)) / total_slots
            cur_frag = self._compute_shannon_entropy(band_slots)

            # Time-weighted averaging
            band_util = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            band_frag = ((last_frag * last_update) + (cur_frag * time_diff)) / self.current_time

            # Update metrics
            self.topology[node1][node2].update({
                util_key: band_util,
                frag_key: band_frag,
                'last_update': self.current_time
            })
   

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(
            1 / self.mean_service_inter_arrival_time
        )
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # generate the bit rate according to the selection adopted
        BitRate = [50, 100, 200]
        bit_rate = random.choice(BitRate)


        self.current_service = Service(
            self.episode_services_processed,
            source=src,
            source_id=src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate,
        )
        self._new_service = True

        

        self.services_processed += 1
        self.episode_services_processed += 1

        # registering statistics about the bit rate requested
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate
        #if self.bit_rate_selection == "discrete":
            #self.bit_rate_requested_histogram[bit_rate] += 1
            #self.episode_bit_rate_requested_histogram[bit_rate] += 1

            # we build the histogram of slots requested assuming the shortest path
            #slots = self.get_number_slots(self.k_shortest_paths[src, dst][0])
            #self.slots_requested_histogram[slots] += 1
            #self.episode_slots_requested_histogram[slots] += 1

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

    # def _get_path_slot_id(self, action: int) -> Tuple[int, int]:
    #     """
    #     Decodes the single action index into the path index and the slot index to be used.

    #     :param action: the single action index
    #     :return: path index and initial slot index encoded in the action
    #     """
    #     path = int(action / self.num_spectrum_resources)
    #     initial_slot = action % self.num_spectrum_resources
    #     return path, initial_slot

    def get_number_slots(self, path: Path, num_bands, band, modulations) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        modulation = self.get_modulation_format(path, num_bands, band, modulations)
        service_bit_rate = self.current_service.bit_rate
        number_of_slots = math.ceil(service_bit_rate / modulation['capacity']) + 1
        return number_of_slots

    def get_shift(slef, band):
        x=0
        y=0
        if band==0:
            x=0
            y=100
        elif band==1:
            x=100
            y=255
        return x , y
    
    def is_path_free(self, path: Path, initial_slot: int, number_slots: int, band) -> bool:
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        if initial_slot_shift + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i+1]
            #use directed edge index
            if np.any(self.topology.graph["available_slots"][
                    ((self.topology[node1][node2]["index"]) +
                    (self.topology.number_of_edges() * band)),
                    initial_slot_shift : initial_slot_shift + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, path: Path, band):
        x = self.get_shift(band)[0]
        y = self.get_shift(band)[1]
        available_slots = functools.reduce(
            np.multiply,
            self.topology.graph["available_slots"][[
                            ((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) + 
                            (self.topology.number_of_edges() * band))
                            for i in range(len(path.node_list) - 1)], 
                            x:y])

        return available_slots

    def rle(inarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path, num_bands, band, modulations):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], band)
        

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], num_bands, band, modulations)
        

        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[: self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def get_available_blocks_FLF(self, path, num_bands, band, modulations):
        """
        Gets available blocks supporting both first-fit and last-fit policies.
        For j=1, returns the first and last valid block in the spectrum.
        
        Returns:
            Dictionary containing:
            - first_fit: (initial_indices, lengths) for first j blocks
            - last_fit: (initial_indices, lengths) for last j blocks
        """
        # Get available slots across the whole path
        available_slots = self.get_available_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], band)
        
        # Get number of slots needed
        slots = self.get_number_slots(
            self.k_shortest_paths[self.current_service.source, self.current_service.destination][path], 
            num_bands, 
            band, 
            modulations
        )
        
        # Get contiguous blocks using run-length encoding
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)
        
        # Get indices of available blocks (where value is 1)
        available_indices = np.where(values == 1)[0]
        
        # Get indices of blocks with sufficient size
        sufficient_indices = np.where(lengths >= slots)[0]
        
        # Get indices that are both available and sufficient
        valid_indices = np.intersect1d(available_indices, sufficient_indices)
        
        if len(valid_indices) == 0:
            # No valid blocks found
            return {
                'first_fit': (np.array([]), np.array([])),
                'last_fit': (np.array([]), np.array([]))
            }
        
        # Get first j blocks (first-fit)
        first_j = valid_indices[:self.j]
        first_fit = (initial_indices[first_j], lengths[first_j])
        
        # Get last j blocks (last-fit)
        last_j = valid_indices[-self.j:]
        last_fit = (initial_indices[last_j], lengths[last_j])
        
        return {'first_fit': first_fit, 'last_fit': last_fit}
        

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.path.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = RMSAEnv.rle(
                self.topology.graph["available_slots"][
                    self.topology[n1][n2]["index"], :
                ]
            )
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += (
                    lambda_max - lambda_min
                )  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSAEnv.rle(
                    self.topology.graph["available_slots"][
                        self.topology[n1][n2]["index"], lambda_min:lambda_max
                    ]
                )
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (
                self.topology.number_of_edges() / sum_unused_spectrum_blocks
            )
        else:
            cur_spectrum_compactness = 1.0

        return cur_spectrum_compactness

    def calculate_MF(self, modulations, length):
        for i in range(len(modulations) - 1):
            if length > modulations[i + 1]['max_reach']:
                if length <= modulations[i]['max_reach']:
                    return modulations[i]
        return modulations[len(modulations) - 1]
    
    def get_modulation_format(self, path: Path, num_bands, band, modulations):
        length= path.length
        if num_bands == 1: # C band
            modulation_format = self.calculate_MF(modulations, length)
        elif num_bands == 2: # C + L band
            if band == 0: # C band
                modulation_format = self.calculate_MF(modulations, length)
            elif band == 1: # L band
                modulation_format = self.calculate_MF(modulations, length)

        return modulation_format 

    '''
        Modluation format
    '''
    #[BPSK, QPSK, 8QAM, 16QAM]
    capacity = [12.5, 25, 37.5, 50]
    modulations = list()
    modulations.append({'modulation': 'BPSK', 'capacity': capacity[0], 'max_reach': 4000})
    modulations.append({'modulation': 'QPSK', 'capacity': capacity[1], 'max_reach': 2000})
    modulations.append({'modulation': '8QAM', 'capacity': capacity[2], 'max_reach': 1000})
    modulations.append({'modulation': '16QAM', 'capacity': capacity[3], 'max_reach': 500})




def shortest_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    num_slots = env.get_number_slots(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ][0]
    )
    for initial_slot in range(
        0, env.topology.graph["num_spectrum_resources"] - num_slots
    ):
        if env.is_path_free(
            env.k_shortest_paths[
                env.current_service.source, env.current_service.destination
            ][0],
            initial_slot,
            num_slots,
        ):
            return (0, initial_slot)
    return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])


def shortest_available_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    for idp, path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                return (idp, initial_slot)
    return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])


def least_loaded_path_first_fit(env: RMSAEnv) -> Tuple[int, int]:
    max_free_slots = 0
    action = (
        env.topology.graph["k_paths"],
        env.topology.graph["num_spectrum_resources"],
    )
    for idp, path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = (idp, initial_slot)
                    max_free_slots = free_slots
                break  # breaks the loop for the initial slot
    return action


class SimpleMatrixObservation(gym.ObservationWrapper):
    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        max_node = max(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate(
            (
                source_destination_tau.reshape(
                    (1, np.prod(source_destination_tau.shape))
                ),
                spectrum_obs.reshape((1, np.prod(spectrum_obs.shape))),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):
    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(
            self.env.k_paths + self.env.reject_action
        )
        self.observation_space = env.observation_space

    def action(self, action) -> Tuple[int, int]:
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(
                self.env.k_shortest_paths[
                    self.env.current_service.source,
                    self.env.current_service.destination,
                ][action]
            )
            for initial_slot in range(
                0, self.env.topology.graph["num_spectrum_resources"] - num_slots
            ):
                if self.env.is_path_free(
                    self.env.k_shortest_paths[
                        self.env.current_service.source,
                        self.env.current_service.destination,
                    ][action],
                    initial_slot,
                    num_slots,
                ):
                    return (action, initial_slot)
        return (
            self.env.topology.graph["k_paths"],
            self.env.topology.graph["num_spectrum_resources"],
        )

    def step(self, action):
        return self.env.step(self.action(action))
