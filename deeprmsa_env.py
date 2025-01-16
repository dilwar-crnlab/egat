from typing import Tuple

import gym
import gym.spaces
import numpy as np
import networkx as nx

from .rmsa_env import RMSAEnv



from .optical_network_env import OpticalNetworkEnv

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        num_bands,
        topology=None,
        j=2,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        k_paths=5,
        allow_rejection=False,
    ):
        super().__init__(
            num_bands=num_bands,
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            k_paths=k_paths,
            allow_rejection=allow_rejection,
            reset=False,
        )

        self.j = j

        # New action space for (path, band, fit) selection
        total_actions = self.k_paths * self.num_bands * self.j + self.reject_action # 2 fits per band
        self.action_space = gym.spaces.Discrete(total_actions)

        max_path_length = self.get_max_path_length() #max number of edges in a path
        #print("Max path length",max_path_length)
        # New observation space for GAT features
        num_nodes = self.topology.number_of_nodes()

        # Transform topology once during initialization
        self.transformed_topology = self.transform_topology()
    

        num_edge_features = self.k_paths * self.get_max_path_length() * 16
        num_service_info = 2 * self.topology.number_of_nodes() + 1  # Source-destination + bit rate

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_edge_features + num_service_info,),
            dtype=np.float32
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        

        self.reset(only_episode_counters=False)




    

    def compute_adjacency_matrix(self, service):
        """
        Compute adjacency matrix for nodes in candidate paths and their neighbors.
        """
        # Get relevant nodes from paths and neighbors
        node_features = self.compute_node_features(service)
        relevant_nodes = list(node_features.keys())
        
        # Create mapping of nodes to indices
        node_to_idx = {node: idx for idx, node in enumerate(relevant_nodes)}
        num_nodes = len(relevant_nodes)
        
        # Initialize adjacency matrix
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Fill adjacency matrix based on original topology
        for i, node1 in enumerate(relevant_nodes):
            for j, node2 in enumerate(relevant_nodes):
                if self.topology.has_edge(node1, node2):
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
        
        return adjacency_matrix, node_to_idx

    def generate_node_features(self, simplified_topology, service):
        """Generate node features using current edge features."""
        # Get edge features using existing method
        path_features = self.compute_path_edge_features(service)
        
        # Initialize node features dictionary
        node_features = {}
        paths = self.k_shortest_paths[service.source, service.destination]
        
        # For each path, create its feature matrix
        for path_idx in range(len(paths)):
            if path_idx in path_features:
                path = paths[path_idx]
                # For each edge in path
                for i in range(len(path.node_list) - 1):
                    node1, node2 = path.node_list[i], path.node_list[i + 1]
                    link_id = self.topology[node1][node2]['index']
                    # Use the already computed edge features as node features
                    node_features[link_id] = path_features[path_idx][i]
        
        return node_features

    def get_max_path_length(self):
        """
        Computes maximum number of edges in any path from k-shortest paths.
        """
        max_length = 0
        
        # Check all source-destination pairs
        for source in self.topology.nodes():
            for dest in self.topology.nodes():
                if source != dest:
                    # Get k-shortest paths for this pair
                    paths = self.k_shortest_paths[source, dest]
                    
                    # Check length of each path
                    for path in paths:
                        num_edges = len(path.node_list) - 1  # number of edges = nodes - 1
                        max_length = max(max_length, num_edges)
                        
        return max_length



    # def observation(self):
    #     """
    #     Returns observation with node features, edge features, and topology information.
    #     """
    #     # Get node features and adjacency information
    #     node_features = self.compute_node_features(self.current_service)
    #     adjacency_matrix, node_to_idx = self.compute_adjacency_matrix(self.current_service)
        
    #     # Get edge features for paths and neighbors
    #     path_edge_features, all_edge_features = self.compute_path_edge_features(self.current_service)
        
    #     # Convert node_features dict to tensor format
    #     node_feature_tensor = []
    #     for node in sorted(node_features.keys()):
    #         node_feature_tensor.append([node_features[node]['betweenness']])
    #     node_feature_tensor = np.array(node_feature_tensor)
        
    #     # Create service info tensor
    #     service_info = np.zeros(29)
    #     service_info[self.current_service.source_id] = 1
    #     service_info[self.topology.number_of_nodes() + self.current_service.destination_id] = 1
    #     service_info[-1] = self.current_service.bit_rate / 200
        
    #     # Flatten and concatenate features
    #     flattened_observation = np.concatenate([
    #         np.array(list(path_edge_features.values())).flatten(),
    #         node_feature_tensor.flatten(),
    #         service_info
    #     ])
        
    #     # Store topology information
    #     self.current_topology_info = {
    #         'adjacency_matrix': adjacency_matrix,
    #         'node_to_idx': node_to_idx,
    #         'path_edge_features': path_edge_features,
    #         'all_edge_features': all_edge_features
    #     }
        
    #     return flattened_observation

    def observation(self):
        """
        Returns observation with node features, edge features, and service info.
        """
        #print(self.current_service.source_id, self.current_service.destination_id, self.current_service.bit_rate)
        # Get path-wise edge features
        path_edge_features = self.compute_path_edge_features(self.current_service)
        
        # Get path-wise node features
        path_node_features = self.compute_node_features(self.current_service)
        
        # Create service info tensor
        service_info = np.zeros(2 * self.topology.number_of_nodes() + 1)
        service_info[self.current_service.source_id] = 1  # source node
        service_info[self.topology.number_of_nodes() + self.current_service.destination_id] = 1  # destination node
        service_info[-1] = self.current_service.bit_rate / 200  # normalized bit rate
        
        # Flatten edge features with fixed size
        edge_features_list = []
        max_path_length = self.get_max_path_length()
        for path_idx in range(self.k_paths):
            if path_idx in path_edge_features:
                # Get features and pad if needed
                path_features = path_edge_features[path_idx]['edge_features']
                # Ensure path_features has shape [max_path_length, 16]
                if path_features.shape[0] < max_path_length:
                    padding = np.zeros((max_path_length - path_features.shape[0], 16))
                    path_features = np.vstack([path_features, padding])
                edge_features_list.append(path_features.flatten())
            else:
                # Add zero features for missing path
                edge_features_list.append(np.zeros(max_path_length * 16))
        
        # Flatten node features with fixed size
        node_features_list = []
        for path_idx in range(self.k_paths):
            if path_idx in path_node_features:
                # Extract betweenness values for nodes in this path
                path_nodes = list(path_node_features[path_idx]['nodes'])
                node_feats = [path_node_features[path_idx]['node_features'][node]['betweenness'] 
                            for node in path_nodes]
                # Pad to max_path_length
                while len(node_feats) < max_path_length:
                    node_feats.append(0.0)
                # Trim if longer than max_path_length
                #node_feats = node_feats[:max_path_length]
                #node_features_list.append(node_feats)
                node_features_list.append(np.array(node_feats))
            else:
                node_features_list.append(np.zeros(max_path_length))
        
        # Concatenate all features
        print("Edge features shape:", edge_features_list)
        flattened_edge_features = np.concatenate(edge_features_list)
        flattened_node_features = np.concatenate(node_features_list)
        
        # Print shapes for debugging
        print("Edge features shape:", flattened_edge_features.shape)

        # print("Node features shape:", flattened_node_features.shape)
        # print("Service info shape:", service_info.shape)
        
        observation = np.concatenate([
            flattened_edge_features,
            flattened_node_features,
            service_info
        ])
        
        # print("Final observation shape:", observation.shape)
        # print("Observation dtype:", observation.dtype)
        
        return observation


    def step(self, action):
        """Handle GAT-based action selection."""
        path_idx, band, fit_type = self._get_route_band_block_id(action)
        #print(path_idx, band, fit_type)
        
        # Get blocks for selected band
        blocks = self.get_available_blocks_FLF(path_idx, self.num_bands, band, self.modulations)
        #print(blocks)
        
        # Select block based on fit_type
        if fit_type == 0:  # first-fit
            initial_slot = blocks['first_fit'][0][0] if blocks['first_fit'][0].size > 0 else -1
        else:  # last-fit
            initial_slot = blocks['last_fit'][0][0] if blocks['last_fit'][0].size > 0 else -1
        
        if initial_slot >= 0:
            # Try to provision
            result = super().step([path_idx, band, initial_slot])
        else:
            # Reject if no valid block
            result = super().step([self.k_paths, self.num_bands, self.num_spectrum_resources])

        obs, rw, _, info = result
        
        return result



    
    def reward(self, path_idx, band):
        return super().reward(path_idx, band)
        

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)


    
    def _get_route_band_block_id(self, action):
        path_index = action // 4  # 4 combinations per path
        remaining = action % 4
        band = remaining // 2     # 2 fits per band
        fit_type = remaining % 2  # first/last
        return path_index, band, fit_type


