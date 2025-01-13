import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

class EGATLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        
        
        
        
        # Transformation layers
        self.W_H = nn.Linear(node_dim, out_dim)  # Node features
        self.W_E_C = nn.Linear(edge_dim, out_dim)  # C-band edge features
        self.W_E_L = nn.Linear(edge_dim, out_dim)  # L-band edge features
        
        # Initialize attention parameters
        # For concatenated [hidden_dim + hidden_dim + hidden_dim]
        attention_dim = 3 * out_dim  # Three times hidden_dim after transformations
        self.a_C = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        self.a_L = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        self.b_C = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        self.b_L = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize attention parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.a_C, gain=gain)
        nn.init.xavier_uniform_(self.a_L, gain=gain)
        nn.init.xavier_uniform_(self.b_C, gain=gain)
        nn.init.xavier_uniform_(self.b_L, gain=gain)
        
    def edge_attention_block(self, H, E_C, E_L, AE, path_edge_indices):
        batch_size = E_C.size(0)
        M = E_C.size(1)
        
        # Transform edge features
        E_C_trans = self.W_E_C(E_C)
        E_L_trans = self.W_E_L(E_L)
        
        # Initialize attention scores
        C_attention_scores = torch.zeros(batch_size, M, M, device=E_C.device)
        L_attention_scores = torch.zeros(batch_size, M, M, device=E_C.device)
        
        # Process each path edge
        for p in path_edge_indices:
            # Get all neighbors (both path and non-path)
            neighbor_edges = torch.where(AE[0,p] > 0)[0]
            
            for q in neighbor_edges:
                # Get shared node features
                h_pq = self._get_shared_node_features(H, p, q)
                
                # C-band attention
                c_concat = torch.cat([
                    E_C_trans[:,p],
                    E_C_trans[:,q],
                    h_pq
                ], dim=-1)
                C_attention_scores[:,p,q] = self.leaky_relu(
                    torch.matmul(c_concat, self.b_C).squeeze(-1)
                )
                
                # L-band attention
                l_concat = torch.cat([
                    E_L_trans[:,p],
                    E_L_trans[:,q],
                    h_pq
                ], dim=-1)
                L_attention_scores[:,p,q] = self.leaky_relu(
                    torch.matmul(l_concat, self.b_L).squeeze(-1)
                )
        
        # Update path edge features
        E_C_new = E_C_trans.clone()
        E_L_new = E_L_trans.clone()
        
        for p in path_edge_indices:
            neighbor_edges = torch.where(AE[0,p] > 0)[0]
            if len(neighbor_edges) > 0:
                # Normalize attention scores
                C_attn = F.softmax(C_attention_scores[:,p,neighbor_edges], dim=-1)
                L_attn = F.softmax(L_attention_scores[:,p,neighbor_edges], dim=-1)
                
                # Update features
                E_C_new[:,p] = torch.sum(
                    C_attn.unsqueeze(-1) * E_C_trans[:,neighbor_edges],
                    dim=1
                )
                E_L_new[:,p] = torch.sum(
                    L_attn.unsqueeze(-1) * E_L_trans[:,neighbor_edges],
                    dim=1
                )
                
        return E_C_new, E_L_new
    
    def node_attention_block(self, H, E_C_star, E_L_star, AH, path_node_indices):
        """
        Node attention block
        """
        batch_size, N, _ = H.size()
        
        # Transform all features to same dimension (hidden_dim)
        H_trans = self.W_H(H)  # [batch_size, N, hidden_dim]
        E_C_trans = self.W_E_C(E_C_star)  # Transform to same hidden_dim
        E_L_trans = self.W_E_L(E_L_star)  # Transform to same hidden_dim
        
        # Initialize attention scores
        C_attention_scores = torch.zeros(batch_size, N, N, device=H.device)
        L_attention_scores = torch.zeros(batch_size, N, N, device=H.device)
        
        # Process each path node
        for i in path_node_indices:
            neighbor_nodes = torch.where(AH[0,i] > 0)[0]
            
            for j in neighbor_nodes:
                # C-band attention
                c_concat = torch.cat([
                    H_trans[:,i],           # [batch_size, hidden_dim]
                    H_trans[:,j],           # [batch_size, hidden_dim]
                    E_C_trans[:,i,j]        # [batch_size, hidden_dim]
                ], dim=-1)                  # Result: [batch_size, 3*hidden_dim]
                
                C_attention_scores[:,i,j] = self.leaky_relu(
                    torch.matmul(c_concat, self.a_C)
                ).squeeze(-1)
                
                # L-band attention
                l_concat = torch.cat([
                    H_trans[:,i],
                    H_trans[:,j],
                    E_L_trans[:,i,j]
                ], dim=-1)
                
                L_attention_scores[:,i,j] = self.leaky_relu(
                    torch.matmul(l_concat, self.a_L)
                ).squeeze(-1)
        
        # Initialize outputs
        H_C_new = H_trans.clone()
        H_L_new = H_trans.clone()
        Hm_C = torch.zeros_like(H_trans)
        Hm_L = torch.zeros_like(H_trans)
        
        # Update features for path nodes
        for i in path_node_indices:
            neighbor_nodes = torch.where(AH[0,i] > 0)[0]
            if len(neighbor_nodes) > 0:
                C_attn = F.softmax(C_attention_scores[:,i,neighbor_nodes], dim=-1)
                L_attn = F.softmax(L_attention_scores[:,i,neighbor_nodes], dim=-1)
                
                # Update node features
                H_C_new[:,i] = torch.sum(C_attn.unsqueeze(-1) * H_trans[:,neighbor_nodes], dim=1)
                H_L_new[:,i] = torch.sum(L_attn.unsqueeze(-1) * H_trans[:,neighbor_nodes], dim=1)
                
                # Update edge-integrated features
                Hm_C[:,i] = torch.sum(
                    C_attn.unsqueeze(-1) * (H_trans[:,neighbor_nodes] * E_C_trans[:,i,neighbor_nodes]),
                    dim=1
                )
                Hm_L[:,i] = torch.sum(
                    L_attn.unsqueeze(-1) * (H_trans[:,neighbor_nodes] * E_L_trans[:,i,neighbor_nodes]),
                    dim=1
                )
        
        return H_C_new, H_L_new, Hm_C, Hm_L

    def forward(self, H, E_C, E_L, AH, AE, ME, MH, path_node_indices, path_edge_indices):
        # Transform edge features to adjacency form
        E_C_star = self._transform_edge_features(E_C, ME)
        E_L_star = self._transform_edge_features(E_L, ME)
        
        # Node attention
        H_C_new, H_L_new, Hm_C, Hm_L = self.node_attention_block(
            H, E_C_star, E_L_star, AH, path_node_indices
        )
        
        # Edge attention
        E_C_new, E_L_new = self.edge_attention_block(
            H, E_C, E_L, AE, path_edge_indices
        )
        
        # Combine features
        H_new = (H_C_new + H_L_new) / 2
        Hm = (Hm_C + Hm_L) / 2
        
        return H_new, E_C_new, E_L_new, Hm
    

    def _transform_edge_features(self, E, ME):
        """
        Transform edge features using edge mapping matrix ME
        Args:
            E: Edge features [batch_size, M, Fe]  # M = 45 (total edges)
            ME: Edge mapping matrix [batch_size, N*N, M]  # N*N = 25, M = 45
        """
        batch_size = E.size(0)
        M = E.size(1)
        Fe = E.size(2)
        
        # Transform edge features
        E_transformed = torch.bmm(ME, E)  # [batch_size, N*N, Fe]
        
        # Reshape to adjacency form
        N = int(torch.sqrt(torch.tensor(ME.size(1))))
        E_star = E_transformed.view(batch_size, N, N, Fe)
        
        return E_star

    def _get_shared_node_features(self, H, edge1, edge2):
        """
        Get features of node shared between two edges
        """
        batch_size = H.size(0)
        node_dim = H.size(-1)
        
        # Create empty tensor for shared features
        shared_features = torch.zeros(batch_size, node_dim, device=H.device)
        
        # Get nodes for each edge
        edge1_start = (edge1 // (H.size(1)-1)) * H.size(1)
        edge1_end = edge1_start + 1
        edge2_start = (edge2 // (H.size(1)-1)) * H.size(1)
        edge2_end = edge2_start + 1
        
        # Find shared node
        if edge1_start == edge2_start:
            shared_features = H[:, edge1_start]
        elif edge1_start == edge2_end:
            shared_features = H[:, edge1_start]
        elif edge1_end == edge2_start:
            shared_features = H[:, edge1_end]
        elif edge1_end == edge2_end:
            shared_features = H[:, edge1_end]
            
        return shared_features
    


class EGATNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=2, num_heads=8):
        super().__init__()
        
        # Stack of EGAT layers
        self.layers = nn.ModuleList([
            EGATLayer(
                node_dim if i==0 else hidden_dim,
                edge_dim if i==0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Multi-scale merge parameters
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.merge_weights = nn.Parameter(
            torch.FloatTensor(num_heads, num_layers)
        )
        
        # Output layers
        service_info_size = 29
        combined_size = hidden_dim + service_info_size
        
        self.action_head = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20)  # 5 paths * 2 bands * 2 fits
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.merge_weights)
    
    def multi_scale_merge(self, H_list, Hm_list):
        """
        Merge features from different layers and heads
        """
        batch_size = H_list[0].size(0)
        
        # Initialize merged features
        merged_features = []
        
        # For each attention head
        for k in range(self.num_heads):
            head_features = []
            
            # Merge features from each layer
            for l in range(self.num_layers):
                # Combine node and edge-integrated features
                layer_features = H_list[l] + Hm_list[l]
                
                # Weight by merge parameter
                weighted_features = self.merge_weights[k,l] * layer_features
                head_features.append(weighted_features)
            
            # Stack layer features
            head_output = torch.stack(head_features, dim=1)
            merged_features.append(head_output)
        
        # Average over heads
        merged = torch.mean(torch.stack(merged_features), dim=0)
        
        # Global mean pooling
        merged = torch.mean(merged, dim=1)
        
        return merged
    
    def forward(self, H, E_C, E_L, AH, AE, ME, MH, service_info, path_node_indices, path_edge_indices):
        # Store features from each layer
        H_list = []
        Hm_list = []
        
        # Forward through EGAT layers
        current_H = H
        current_E_C = E_C
        current_E_L = E_L
        
        for layer in self.layers:
            current_H, current_E_C, current_E_L, current_Hm = layer(
                current_H,
                current_E_C,
                current_E_L,
                AH, AE, ME, MH,
                path_node_indices,
                path_edge_indices
            )
            
            H_list.append(current_H)
            Hm_list.append(current_Hm)
        
        # Multi-scale merge
        merged_features = self.multi_scale_merge(H_list, Hm_list)
        
        # Combine with service info
        combined = torch.cat([merged_features, service_info], dim=1)
        
        # Get action logits and value
        action_logits = self.action_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Constants
        self.k_paths = 5
        self.max_path_length = 9
        self.edge_feature_dim = 16
        self.node_feature_dim = 1
        
        # Calculate dimensions
        self.num_edge_features = self.k_paths * self.max_path_length * self.edge_feature_dim
        self.num_node_features = self.k_paths * self.node_feature_dim
        self.num_service_info = 29
        
        # Initialize EGAT network
        # self.egat_network = EGATNetwork(
        #     node_dim=self.node_feature_dim,
        #     edge_dim=self.edge_feature_dim//2,  # Split between C and L bands
        #     hidden_dim=64,
        #     num_layers=2,
        #     num_heads=8
        # )
        # Initialize EGAT network
        self.egat_network = EGATNetwork(
            node_dim=self.node_feature_dim,
            edge_dim=8,  # Each band has 8 features
            hidden_dim=64,
            num_layers=2,
            num_heads=8
        )
        
        # Initialize matrices
        self.register_buffer('node_indices', None)
        self.register_buffer('edge_indices', None)
        self.register_buffer('AH', None)
        self.register_buffer('AE', None)
        self.register_buffer('ME', None)
        self.register_buffer('MH', None)
    
    def _process_observation(self, obs):
        """
        Process raw observation into EGAT inputs
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Debug dimensions
        print("Input observation shape:", obs.shape)
        
        # 1. Extract and reshape edge features
        edge_features = obs[:, :self.num_edge_features].view(
            batch_size, 
            self.k_paths,
            self.max_path_length, 
            self.edge_feature_dim
        )
        print("Edge features shape:", edge_features.shape)
        
        # Reshape edge features for attention computation
        c_band_features = edge_features[...,:8].reshape(batch_size, -1, 8)   # [batch, k_paths*max_path_length, 8]
        l_band_features = edge_features[...,8:].reshape(batch_size, -1, 8)   # [batch, k_paths*max_path_length, 8]
        
        print("C-band features shape:", c_band_features.shape)
        print("L-band features shape:", l_band_features.shape)
        
        # 2. Extract node features
        node_features = obs[:, 
                        self.num_edge_features:self.num_edge_features+self.num_node_features
                        ].view(batch_size, self.k_paths, self.node_feature_dim)
        print("Node features shape:", node_features.shape)
        
        # 3. Extract service info
        service_info = obs[:, -self.num_service_info:]
        print("Service info shape:", service_info.shape)
        
        # 4. Get path indices and build matrices if not already done
        if self.node_indices is None:
            path_node_indices, path_edge_indices = self._get_path_indices(edge_features)
            self.node_indices = torch.tensor(path_node_indices, device=device)
            self.edge_indices = torch.tensor(path_edge_indices, device=device)
            
            self.AH, self.AE, self.ME, self.MH = self._build_matrices(
                edge_features,
                self.node_indices,
                self.edge_indices
            )
            print("AH shape:", self.AH.shape)
            print("AE shape:", self.AE.shape)
            print("ME shape:", self.ME.shape)
            print("MH shape:", self.MH.shape)
        
        return (node_features, c_band_features, l_band_features, service_info)

    def _get_path_indices(self, edge_features):
        """
        Get indices of nodes and edges in paths
        """
        node_set = set()
        edge_list = []
        
        # For each path
        for path_idx in range(self.k_paths):
            # Get valid nodes in current path
            valid_edges = torch.any(edge_features[0,path_idx] != 0, dim=-1)
            nodes_in_path = torch.nonzero(valid_edges).squeeze(-1)
            
            for node_idx in nodes_in_path:
                node_set.add(path_idx)
            
            # Get valid edges
            for i in range(len(nodes_in_path) - 1):
                edge_idx = path_idx * (self.max_path_length - 1) + i
                edge_list.append(edge_idx)
        
        return list(node_set), edge_list

    def _build_matrices(self, edge_features, node_indices, edge_indices):
        """
        Build adjacency and mapping matrices
        """
        batch_size = edge_features.size(0)
        total_edges = self.k_paths * self.max_path_length  # 5 * 9 = 45 edges total
        device = edge_features.device
        
        # 1. Node adjacency matrix (AH)
        AH = torch.zeros(batch_size, self.k_paths, self.k_paths, device=device)
        
        for i in node_indices:
            for j in range(self.k_paths):
                path_i = edge_features[:,i,:,:]
                path_j = edge_features[:,j,:,:]
                shared = torch.any(torch.all(path_i.unsqueeze(2) == path_j.unsqueeze(1), dim=-1))
                if shared:
                    AH[:,i,j] = 1
                    AH[:,j,i] = 1
        
        # Add self-loops
        AH.diagonal(dim1=1, dim2=2).fill_(1)
        
        # 2. Edge adjacency matrix (AE)
        AE = torch.zeros(batch_size, total_edges, total_edges, device=device)
        
        # Connect edges sharing nodes in paths
        for i in range(total_edges):
            path_i = i // self.max_path_length
            pos_i = i % self.max_path_length
            
            for j in range(total_edges):
                path_j = j // self.max_path_length
                pos_j = j % self.max_path_length
                
                # Check if edges are connected (share a node)
                if pos_i + 1 == pos_j or pos_i == pos_j + 1:
                    if path_i == path_j:  # Same path
                        AE[:,i,j] = 1
                        AE[:,j,i] = 1
        
        # Add self-loops
        AE.diagonal(dim1=1, dim2=2).fill_(1)
        
        # 3. Edge mapping matrix (ME)
        ME = torch.zeros(batch_size, self.k_paths * self.k_paths, total_edges, device=device)
        
        # Map edges to their paths
        for i in range(total_edges):
            path_idx = i // self.max_path_length
            ME[:,path_idx*self.k_paths:(path_idx+1)*self.k_paths,i] = 1
        
        # 4. Node mapping matrix (MH)
        MH = torch.zeros(batch_size, total_edges, self.k_paths, device=device)
        
        # Map edges to their path nodes
        for i in range(total_edges):
            path_idx = i // self.max_path_length
            MH[:,i,path_idx] = 1
        
        return AH, AE, ME, MH

    def forward(self, obs, deterministic=False):
        """
        Policy forward pass
        """
        # Process observation
        node_features, c_band_features, l_band_features, service_info = self._process_observation(obs)
        
        # Forward through EGAT network
        action_logits, values = self.egat_network(
            node_features,
            c_band_features,
            l_band_features,
            self.AH,
            self.AE,
            self.ME,
            self.MH,
            service_info,
            self.node_indices,
            self.edge_indices
        )
        
        # Get action distribution
        distribution = self.get_distribution(action_logits)
        
        # Sample action
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
            
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for training
        """
        node_features, c_band_features, l_band_features, service_info = self._process_observation(obs)
        
        action_logits, values = self.egat_network(
            node_features,
            c_band_features,
            l_band_features,
            self.AH,
            self.AE,
            self.ME,
            self.MH,
            service_info,
            self.node_indices,
            self.edge_indices
        )
        
        distribution = self.get_distribution(action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def get_distribution(self, action_logits):
        """
        Convert logits to categorical distribution
        """
        action_probs = F.softmax(action_logits, dim=-1)
        return Categorical(probs=action_probs)
    


import os
import pickle
import numpy as np
import torch
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current reward mean from all previous logs
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # Save the model if mean reward is better
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

def main():
    # Create directories
    log_dir = "./tmp/deeprmsa-egat-a2c/"
    os.makedirs(log_dir, exist_ok=True)

    # Load topology
    topology_name = 'nsfnet_chen_link_span'
    k_paths = 5
    with open(f'../topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
        topology = pickle.load(f)

    # Environment setup
    env_args = dict(
        num_bands=2,
        topology=topology, 
        seed=10,
        allow_rejection=False,
        mean_service_holding_time=5,
        mean_service_inter_arrival_time=0.1,
        k_paths=5,
        episode_length=100
    )

    # Monitor keywords for logging
    monitor_info_keywords = (
        "service_blocking_rate",
        "episode_service_blocking_rate",
        "bit_rate_blocking_rate",
        "episode_bit_rate_blocking_rate"
    )

    # Create and wrap environment
    env = gym.make('DeepRMSA-v0', **env_args)
    env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)

    # Initialize A2C with custom EGAT policy
    model = A2C(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="./tb/EGAT-DeepRMSA-A2C/",
        learning_rate=7e-4,
        n_steps=5,           # Number of steps before updating
        gamma=0.99,          # Discount factor
        gae_lambda=1.0,      # GAE parameter
        ent_coef=0.01,       # Entropy coefficient
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        rms_prop_eps=1e-5,   # RMSprop epsilon
        use_rms_prop=True,   # Use RMSprop optimizer
        normalize_advantage=True
    )

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=log_dir,
        verbose=1
    )

    # Train the agent
    total_timesteps = 1000000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="egat_a2c_run"
        )
        
        # Save final model
        model.save(f"{log_dir}/final_model")
        
    except KeyboardInterrupt:
        print("Training interrupted! Saving current model...")
        model.save(f"{log_dir}/interrupted_model")

    # Test the trained model
    print("Testing trained model...")
    obs = env.reset()
    done = False
    total_reward = 0
    num_steps = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        num_steps += 1

        if num_steps % 100 == 0:
            print(f"Step {num_steps}, Current blocking rate: {info['service_blocking_rate']:.4f}")

    print(f"Testing completed!")
    print(f"Final service blocking rate: {info['service_blocking_rate']:.4f}")
    print(f"Final bit rate blocking rate: {info['bit_rate_blocking_rate']:.4f}")

if __name__ == "__main__":
    main()