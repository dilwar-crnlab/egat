from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv

from .optical_network_env import OpticalNetworkEnv

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        num_bands,
        topology=None,
        j=1,
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
        #shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths * self.num_bands

      
        # Define components of the observation space
        source_dest_size = 2 * self.topology.number_of_nodes()  # source and destination one-hot vectors
        
        # Link features per band:
        # 1. utilization
        # 2. fragmentation
        # 3. connections_score
        # 4. avg_osnr_margin
        # 5. available_slots_ratio
        # 6. available_blocks
        # 7. avg_block_size_ratio
        # 8. largest_block_ratio
        features_per_band = 8
        
        # General link features:
        # 1. spans_score
        general_link_features = 1
        
        # Total features per link = general features + (features per band * num bands)
        features_per_link = general_link_features + (features_per_band * self.num_bands)
        
        # Total size for link features
        link_features_size = self.topology.number_of_edges() * features_per_link

        shape = (
            2 * self.topology.number_of_nodes() +     # source_dest (2 one-hot vectors)
            self.topology.number_of_nodes() +         # node betweenness
            self.k_paths +                            # slots_per_path
            (self.k_paths * 6 * self.num_bands) +     # spectrum distribution (6 features per band per path)
            (link_features_size) +
            (self.topology.number_of_nodes() * self.topology.number_of_nodes()) +  # node adjacency matrix
            (self.topology.number_of_edges() * self.topology.number_of_edges())  # edge adjacency matrix
        )
        print("shape", shape)
    
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(shape,), 
            dtype=np.uint8
        )

        #self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths  * self.num_bands * self.j + self.reject_action)
        print(self.k_paths  * self.num_bands * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters=False)

    def step(self, action: int):
        parent_step_result = None
        valid_action = False
        #print("Action ", action)

        if action < self.k_paths * self.j * self.num_bands:  # action is for assigning a route
            valid_action = True
            route, band, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route, self.num_bands, band, self.modulations)
            slots = self.get_number_slots(self.k_shortest_paths[self.current_service.source, self.current_service.destination][route], self.num_bands, band, self.modulations)
            if block < len(initial_indices):
                parent_step_result = super().step(
                    [route, band, initial_indices[block]])
            else:
                parent_step_result = super().step(
                    [self.k_paths, self.num_bands, self.num_spectrum_resources])
        else:
            parent_step_result = super().step(
                [self.k_paths, self.num_bands, self.num_spectrum_resources])

        obs, rw, _, info = parent_step_result
        info['slots'] = slots if valid_action else -1
        return parent_step_result


    def observation(self):
        """
        Create the complete observation vector with all components.
        """
        src = self.current_service.source
        dst = self.current_service.destination
        
        # 1. Source-destination encoding
        source_one_hot = np.zeros(self.topology.number_of_nodes())
        dest_one_hot = np.zeros(self.topology.number_of_nodes())
        source_one_hot[self.topology.nodes[src]['index']] = 1
        dest_one_hot[self.topology.nodes[dst]['index']] = 1
        source_dest_encoding = np.concatenate([source_one_hot, dest_one_hot])

        #print("Original source_dest in observation:", source_dest_encoding)
        
        # 2. Slots per path
        slots_per_path = np.zeros(self.k_paths)
        paths = self.k_shortest_paths[(src, dst)]
        for i, path in enumerate(paths):
            slots_per_path[i] = self.get_number_slots(path, self.num_bands, 0, self.modulations)
        
        # 3. Spectrum distribution (6 features per band per path)
        spectrum_features = []
        for path_idx, path in enumerate(paths):
            for band in range(self.num_bands):
                available_slots = self.get_available_slots(path, band)
                #print("available_slots", available_slots)
                num_slots = self.get_number_slots(path, self.num_bands, band, self.modulations)
                initial_indices, lengths = self.get_available_blocks(path_idx, self.num_bands, band, self.modulations)
                
                N_FS = np.sum(available_slots)
                N_FSB = len(lengths)
                N_FSB_prime = len([l for l in lengths if l >= num_slots])
                I_start = initial_indices[0] if lengths else -1
                S_first = lengths[0] if lengths else -1
                S_FSB = np.mean(lengths) if lengths else 0
                
                spectrum_features.extend([N_FS, N_FSB, N_FSB_prime, I_start, S_first, S_FSB])
        
        # 4. Link features
        link_band_features = self.get_band_aware_link_features(self.k_shortest_paths, self.current_service)
        link_features_vector = []
        for link_idx in range(self.topology.number_of_edges()):
            if link_idx in link_band_features:
                link_features_vector.append(link_band_features[link_idx]['spans_score'])
            else:
                link_features_vector.append(0)
                
            for band in range(self.num_bands):
                if link_idx in link_band_features and band in link_band_features[link_idx]:
                    features = link_band_features[link_idx][band]
                    link_features_vector.extend([
                        features['utilization'],
                        features['fragmentation'],
                        features['connections_score'],
                        features['avg_osnr_margin'],
                        features['available_slots_ratio'],
                        features['available_blocks'],
                        features['avg_block_size_ratio'],
                        features['largest_block_ratio']
                    ])
                else:
                    link_features_vector.extend([0] * 8)
        
        # 5. Node adjacency matrix
        node_adj = self.global_node_adj_matrix[(src, dst)]
        merged_node_adj = np.sum(list(node_adj.values()), axis=0)
        merged_node_adj = np.where(merged_node_adj > 0, 1, 0)
        node_adj_flat = merged_node_adj.flatten()
        
        # 6. Edge adjacency matrix
        edge_adj = self.global_edge_adj_matrix[(src, dst)]
        merged_edge_adj = np.sum(list(edge_adj.values()), axis=0)
        merged_edge_adj = np.where(merged_edge_adj > 0, 1, 0)
        edge_adj_flat = merged_edge_adj.flatten()


        # 7. Get betweenness centrality for all nodes
        betweenness_features = self.get_node_betweenness_features()
        
        # Combine all components
        observation = np.concatenate([
            source_dest_encoding,
            slots_per_path,
            spectrum_features,
            link_features_vector,
            betweenness_features,    # Add betweenness features
            node_adj_flat,
            edge_adj_flat
        ]).reshape(self.observation_space.shape)
        print("Obs shape in observation mthod", observation.shape)
        print("Final observation before return:", observation[:28])  # Print just source-dest part
        return observation


    def get_link_available_slots(self, link_idx, band):

        # Get slot range for band
        if band == 0:  # C-band
            x, y = 0, 100
        else:  # L-band
            x, y = 100, 256
        
        # Get available slots for this link and band
        base_idx = link_idx + (self.topology.number_of_edges() * band)
        return self.topology.graph["available_slots"][base_idx][x:y]

    def get_node_betweenness_features(self):
        """
        Get betweenness centrality values for nodes in candidate paths.
        Values are already normalized between 0 and 1.
        
        Args:
            paths: List of paths between source and destination
        
        Returns:
            Array of betweenness values for all nodes
        """
        betweenness_values = np.zeros(self.topology.number_of_nodes())
        
        # Fill in the betweenness values for all nodes
        for i in range(self.topology.number_of_nodes()):
            node_id = str(i + 1)  # Convert to string to match the dictionary keys
            if node_id in self.node_betweenness:
                betweenness_values[i] = self.node_betweenness[node_id]
        
        return betweenness_values

    def get_band_aware_link_features(self, k_shortest_paths, service):
        """
        Compute link features directly from link spectrum state.
        
        C-band: slots 0-99 (100 slots)
        L-band: slots 100-256 (157 slots)
        """
        src = service.source
        dst = service.destination
        paths = k_shortest_paths[(src, dst)]
        
        # Define band ranges and sizes
        band_ranges = {
            0: (0, 100),     # C-band: 0-99
            1: (100, 257)    # L-band: 100-256
        }
        
        band_sizes = {
            0: 100,  # C-band size
            1: 156   # L-band size
        }
        
        # Get all unique links across candidate paths
        unique_links = set()
        for path in paths:
            unique_links.update(path.link_idx)
            
        # Find max spans considering all topology links
        #max_spans = max(len(link.spans) for link in self.topology.graph['links'].values())
        # Get the maximum number of spans in any single link
        max_spans = max(len(self.topology[node1][node2]["link"].spans) 
                for node1, node2 in self.topology.edges())
        
        # Set max connections as half of band size
        max_connections = {
            0: band_sizes[0] // 2,  # max connections for C-band
            1: band_sizes[1] // 2   # max connections for L-band
        }
        
        # Initialize features dictionary
        link_features = {}
        
        # For each unique link, compute features
        for link_idx in unique_links:
            link_features[link_idx] = {}
            
            # 1. Get general link features - normalize so fewer spans = higher score
            num_spans = self.topology.graph['link_spans'][link_idx]
            #print("Num spans", num_spans, "Link", link_idx)

            link_features[link_idx]['spans_score'] = 1 - (num_spans / max_spans) if max_spans > 0 else 1
            
            # Get spectrum state for this link
            link_spectrum = self.topology.graph['available_spectrum'][link_idx]
            
            
            # 2. Compute band-specific features
            for band in range(self.num_bands):
                # Get band slice
                link_spectrum = self.get_link_available_slots(link_idx, band)
                #print("link_spectrum", link_spectrum)
                start_idx, end_idx = band_ranges[band]
                #band_slots = link_spectrum[start_idx:end_idx]
                total_slots = band_sizes[band]
                
                # Find contiguous blocks
                blocks = []  # Will store size of each available block
                current_block = 0
                
                for slot in link_spectrum:
                    if slot == 1:  # Available slot
                        current_block += 1
                    elif current_block > 0:  # End of block
                        blocks.append(current_block)
                        current_block = 0
                        
                if current_block > 0:  # Add last block if exists
                    blocks.append(current_block)
                
                # Calculate metrics
                N_FS = np.sum(link_spectrum)  # total available FSs
                N_FSB = len(blocks)  # total number of blocks

                # print("N_FS", N_FS)
                # print("N_FSB", N_FSB)
                
                active_connections , avg_osnr_margin = self.count_active_connections_osnr_margin(link_idx, band)
                
                # Calculate EnFM fragmentation
                fragmentation = self.calculate_entropy_fragmentation(link_spectrum)
                
                # Store band-specific features
                link_features[link_idx][band] = {
                    # 1. Spectrum utilization (normalized by band-specific size)
                    'utilization': 1 - (N_FS / total_slots),
                    
                    # 2. Fragmentation (Entropy-based)
                    'fragmentation': fragmentation,
                    
                    # 3. Number of active connections (normalized by half of band size)
                    'connections_score': 1 - (active_connections / max_connections[band]),
                    
                    # 4. Average OSNR margin
                    'avg_osnr_margin': avg_osnr_margin,
                    
                    # 5. Number of available spectrum slots (normalized)
                    'available_slots_ratio': N_FS / total_slots,
                    
                    # 6. Number of available spectrum blocks
                    'available_blocks': N_FSB,
                    
                    # 7. Average size of spectrum blocks (normalized)
                    'avg_block_size_ratio': (np.mean(blocks) / total_slots) if blocks else 0,
                    
                    # 8. Size of largest available block (normalized)
                    'largest_block_ratio': (max(blocks) / total_slots) if blocks else 0
                }
        
        return link_features

    def calculate_entropy_fragmentation(self, slots):
        """
        Calculate Entropy-based Fragmentation Metric (EnFM) as per equation (3).
        
        H2 = -sum(Zq/|N| * ln(Zq/|N|)) for q=1 to Q
        where:
        - Q is total number of contiguous slot blocks
        - Zq is number of slots in qth block
        - |N| is total number of slots
        
        Args:
            slots: Array of 0s and 1s representing occupied/free slots
            
        Returns:
            Fragmentation metric H2
        """
        N = len(slots)  # Total number of slots
        
        # Find contiguous blocks (both occupied and available)
        blocks = []  # Will store size of each block
        current_block_size = 1
        current_state = slots[0]
        
        # Count block sizes
        for i in range(1, N):
            if slots[i] == current_state:
                current_block_size += 1
            else:
                blocks.append(current_block_size)
                current_block_size = 1
                current_state = slots[i]
        
        # Add last block
        blocks.append(current_block_size)
        
        # Calculate entropy using equation (3)
        H2 = 0
        for Zq in blocks:
            ratio = Zq / N
            H2 -= (ratio * np.log(ratio))
            
        return H2


    def get_link(self, link_idx):
        # Iterate through edges to find matching link_idx
        for edge in self.topology.edges():
            if self.topology[edge[0]][edge[1]]['index'] == link_idx:
                return self.topology[edge[0]][edge[1]]
            
    def count_active_connections_osnr_margin(self, link_idx, band):
        """Count number of active connections in a specific band of a link."""
        #print("Link idx", link_idx)
        link = self.get_link(link_idx)
        #print(link)
        node1 = link['link'].node1
        node2 = link['link'].node2
        active_services = self.topology[node1][node2]['running_services']
        #print("Active ", active_services)

        active_connection_count = 0
        avg_osnr_margin = 0
        OSNR_margins = []
        if len(active_services) == 0:
            return active_connection_count, avg_osnr_margin
        if len(active_services):
            #print("Active ", active_services)
            for service in active_services:
                if service.band == band:
                    active_connection_count += 1
            
            for service in active_services:
                if service.band == band and service.OSNR_margin is not None:
                    OSNR_margins.append(service.OSNR_margin)
        avg_osnr_margin = np.mean(OSNR_margins) if avg_osnr_margin else 0

        return active_connection_count , avg_osnr_margin


    def calculate_avg_osnr_margin(self, link_idx, band):
        """Calculate average OSNR margin for active connections in a band."""
        active_services = self.topology[link_idx[0]][link_idx[1]]['running_services']
        margins = []
        for service in active_services:
            if service.band == band and service.OSNR_margin is not None:
                margins.append(service.OSNR_margin)
        return np.mean(margins) if margins else 0





    def get_spectrum_distribution_features(self, k_shortest_paths, service):
        """
        Get spectrum availability distribution for all candidate paths using RMSA env methods.
        
        Args:
            rmsa_env: RMSA environment instance
            k_shortest_paths: Dictionary of k-shortest paths
            service_source: Source node
            service_destination: Destination node
        
        Returns:
            Dictionary of spectrum features for each path
        """
        src = service.source
        dst = service.destination
        paths = k_shortest_paths[(src, dst)]
        slots_per_path = []
        spectrum_features = {} 
        for path_idx, path in enumerate(paths):
           
            num_slots = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            #print("num_slots", num_slots)
            slots_per_path.append(num_slots)

            spectrum_features[path_idx] = {}
            for band in range(self.num_bands):
                available_slots = self.get_available_slots(path, band)         
                num_slots = self.get_number_slots(path, self.num_bands, band, self.modulations)
                initial_indices, lengths = self.get_available_blocks(path_idx, self.num_bands, band, self.modulations)
                # Calculate spectrum features
                N_FS = np.sum(available_slots)  # total available FSs
                N_FSB = len(lengths)  # total number of blocks
                #print(N_FS, N_FSB)
                
                # Count blocks that satisfy bandwidth requirement
                N_FSB_prime = len([l for l in lengths if l >= num_slots])   
                # Get first fit block information that satisfies bandwidth requirement
                for i, (start, length) in enumerate(zip(initial_indices, lengths)):
                    if length >= num_slots:  # Check if block is large enough
                        I_start = start      # Starting index of first valid block
                        S_first = length     # Size of first valid block
                        break
                else:  # No blocks satisfy bandwidth requirement
                    I_start = -1  
                    S_first = -1        
                # Calculate average block size
                S_FSB = np.mean(lengths) if lengths else 0   
                spectrum_features[path_idx][band] = {
                    'N_FS': N_FS,
                    'N_FSB': N_FSB,
                    'N_FSB_prime': N_FSB_prime,
                    'I_start': I_start,
                    'S_first': S_first,
                    'S_FSB': S_FSB,
                }
        return spectrum_features, slots_per_path

    

    def reward(self, band, path_selected):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // (self.j * self.num_bands)
        band  = action // (self.j * self.k_paths)
        block = action % self.j
        return route, band, block


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
