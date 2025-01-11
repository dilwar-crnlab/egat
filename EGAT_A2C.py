

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
import gym
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical



from examples.callback import SaveOnBestTrainingRewardCallback
from IPython.display import clear_output

class EGATLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        
        # Feature transformation matrices (Section 4.1)
        self.W_H = nn.Linear(node_dim, out_dim)  # For node features 
        self.W_E = nn.Linear(edge_dim, out_dim)  # For edge features
        
        # Attention parameters
        self.a = nn.Parameter(torch.FloatTensor(3 * out_dim, 1))  # Node attention
        self.b = nn.Parameter(torch.FloatTensor(3 * out_dim, 1))  # Edge attention
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize attention parameters"""
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.b.data)
    
    def transform_edge_features(self, E, ME):
        """
        Transform edge features using edge mapping matrix ME (Figure 2b)
        Args:
            E: Edge features [batch_size, M, Fe]
            ME: Edge mapping matrix [batch_size, N*N, M]
        Returns:
            E*: Transformed edge features in adjacency form [batch_size, N, N, Fe]
        """
        batch_size = E.size(0)
        N = int(torch.sqrt(torch.tensor(ME.size(1))))  # N*N dimension
        Fe = E.size(-1)
        
        # Reshape ME to N^2 x M as shown in paper
        ME_reshaped = ME.view(batch_size, N*N, -1)
        
        # Transform edge features to adjacency form
        E_transformed = torch.bmm(ME_reshaped, E)  # [batch_size, N*N, Fe]
        
        # Reshape back to N x N x Fe
        E_star = E_transformed.view(batch_size, N, N, Fe)
        
        return E_star
    
    def node_attention_block(self, H, E_star, AH):
        """
        Node attention block (Section 4.2)
        Args:
            H: Node features [batch_size, N, Fh]
            E_star: Edge features in adjacency form [batch_size, N, N, Fe]
            AH: Node adjacency matrix [batch_size, N, N]
        Returns:
            H': Updated node features [batch_size, N, Fh']
            Hm: Edge-integrated node features [batch_size, N, Fh']
        """
        batch_size, N, _ = H.size()
        
        # Apply feature transformations
        H_trans = self.W_H(H)  # WH·H
        
        # Compute attention coefficients
        attention_scores = torch.zeros(batch_size, N, N, device=H.device)
        
        for i in range(N):
            for j in range(N):
                if AH[0,i,j] > 0:  # Check connectivity using adjacency matrix
                    # Concatenate features as per Eq(1)
                    node_edge_concat = torch.cat([
                        H_trans[:,i],
                        H_trans[:,j],
                        E_star[:,i,j]
                    ], dim=-1)
                    
                    # Compute attention coefficient
                    attention_scores[:,i,j] = self.leaky_relu(
                        torch.matmul(node_edge_concat, self.a).squeeze(-1)
                    )
                else:
                    attention_scores[:,i,j] = -9e15  # Mask non-connected nodes
        
        # Apply softmax to normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Update node features (Eq 2)
        H_prime = torch.zeros_like(H_trans)
        Hm = torch.zeros_like(H_trans)  # Edge-integrated features
        
        for i in range(N):
            # Get neighbors using adjacency matrix
            neighbors = torch.where(AH[0,i] > 0)[0]
            
            # Update node features
            H_prime[:,i] = torch.sum(
                attention_weights[:,i,neighbors].unsqueeze(-1) * H_trans[:,neighbors],
                dim=1
            )
            
            # Compute edge-integrated features (Eq 3)
            Hm[:,i] = torch.sum(
                attention_weights[:,i,neighbors].unsqueeze(-1) * 
                (H_trans[:,neighbors] * E_star[:,i,neighbors]),
                dim=1
            )
        
        return H_prime, Hm
    
    def edge_attention_block(self, H, E, AE, MH):
        """
        Edge attention block with graph transformation (Section 4.3, Figure 2c-e)
        Args:
            H: Node features [batch_size, N, Fh]
            E: Edge features [batch_size, M, Fe]
            AE: Edge adjacency matrix [batch_size, M, M]
            MH: Node mapping matrix [batch_size, M, N]
        Returns:
            E': Updated edge features [batch_size, M, Fe']
        """
        batch_size, M, _ = E.size()
        
        # Apply feature transformations
        E_trans = self.W_E(E)  # WE·E
        
        # Compute attention coefficients
        attention_scores = torch.zeros(batch_size, M, M, device=E.device)
        
        for p in range(M):
            for q in range(M):
                if AE[0,p,q] > 0:  # Check connectivity in transformed graph
                    # Get shared node features using node mapping matrix
                    h_pq = torch.matmul(MH[:,p] * MH[:,q].unsqueeze(-1), H).squeeze(1)
                    
                    # Concatenate features as per Eq(4)
                    edge_node_concat = torch.cat([
                        E_trans[:,p],
                        E_trans[:,q],
                        h_pq
                    ], dim=-1)
                    
                    # Compute attention coefficient
                    attention_scores[:,p,q] = self.leaky_relu(
                        torch.matmul(edge_node_concat, self.b).squeeze(-1)
                    )
                else:
                    attention_scores[:,p,q] = -9e15
        
        # Apply softmax to normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Update edge features (Eq 5)
        E_prime = torch.zeros_like(E_trans)
        
        for p in range(M):
            # Get neighbor edges using edge adjacency matrix
            neighbors = torch.where(AE[0,p] > 0)[0]
            
            E_prime[:,p] = torch.sum(
                attention_weights[:,p,neighbors].unsqueeze(-1) * E_trans[:,neighbors],
                dim=1
            )
        
        return E_prime
    
    def forward(self, H, E, AH, AE, ME, MH):
        """
        Complete EGAT layer forward pass
        Args:
            H: Node features [batch_size, N, Fh]
            E: Edge features [batch_size, M, Fe]
            AH: Node adjacency matrix [batch_size, N, N]
            AE: Edge adjacency matrix [batch_size, M, M]
            ME: Edge mapping matrix [batch_size, N*N, M]
            MH: Node mapping matrix [batch_size, M, N]
        Returns:
            H': Updated node features
            E': Updated edge features
            Hm: Edge-integrated node features (for merge layer)
        """
        # Transform edge features to adjacency form (Fig 2b)
        E_star = self.transform_edge_features(E, ME)
        
        # Node attention block
        H_prime, Hm = self.node_attention_block(H, E_star, AH)
        
        # Edge attention block with graph transformation
        E_prime = self.edge_attention_block(H, E, AE, MH)
        
        return H_prime, E_prime, Hm
    


class EGATNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=2, num_heads=8):
        """
        Args:
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of EGAT layers (L in paper)
            num_heads: Number of attention heads (K in paper)
        """
        super().__init__()
        
        # Stack of EGAT layers
        self.layers = nn.ModuleList([
            EGATLayer(
                node_dim if i==0 else hidden_dim,
                edge_dim if i==0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Multi-scale merge parameters (Equation 6 in paper)
        self.merge_weights = nn.Parameter(
            torch.FloatTensor(num_heads, num_layers)
        )
        
        # Merge layer convolution (Section 4.4)
        self.merge_conv = nn.Conv1d(
            in_channels=num_layers * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # For DeepRMSA specific outputs
        service_info_size = 29
        combined_size = hidden_dim + service_info_size
        
        self.action_head = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20)  # 5 paths * 2 bands * 2 fit types
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize merge layer parameters"""
        nn.init.xavier_uniform_(self.merge_weights)
        
    def multi_scale_merge(self, H_list, Hm_list):
        """
        Multi-scale merge strategy (Section 4.4)
        Args:
            H_list: List of node features from each layer [L x batch_size x N x hidden_dim]
            Hm_list: List of edge-integrated features from each layer [L x batch_size x N x hidden_dim]
        Returns:
            H_final: Final node representations [batch_size x N x hidden_dim]
        """
        batch_size = H_list[0].size(0)
        N = H_list[0].size(1)
        
        # Initialize merged features for each head
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
            head_output = torch.stack(head_features, dim=1)  # [batch_size x L x N x hidden_dim]
            merged_features.append(head_output)
        
        # Average over heads
        merged = torch.mean(torch.stack(merged_features), dim=0)
        
        # Apply 1D convolution for final transformation
        merged = merged.permute(0, 3, 1, 2)  # [batch_size x hidden_dim x L x N]
        merged = self.merge_conv(merged.reshape(batch_size * N, self.num_layers, -1))
        merged = merged.reshape(batch_size, N, -1)
        
        return merged
        
    def forward(self, H, E, AH, AE, ME, MH, service_info):
        """
        Complete EGAT network forward pass
        Args:
            H: Initial node features
            E: Initial edge features
            AH: Node adjacency matrix 
            AE: Edge adjacency matrix
            ME: Edge mapping matrix
            MH: Node mapping matrix
            service_info: Service request information
        Returns:
            action_logits: Path and band selection logits
            value: Value function estimate
        """
        # Store features from each layer
        H_list = []
        E_list = []
        Hm_list = []
        
        # Forward through EGAT layers
        current_H, current_E = H, E
        
        for layer in self.layers:
            current_H, current_E, current_Hm = layer(
                current_H, current_E,
                AH, AE, ME, MH
            )
            
            H_list.append(current_H)
            E_list.append(current_E)
            Hm_list.append(current_Hm)
        
        # Multi-scale merge
        merged_features = self.multi_scale_merge(H_list, Hm_list)
        
        # Global average pooling
        graph_features = torch.mean(merged_features, dim=1)
        
        # Combine with service info
        combined = torch.cat([graph_features, service_info], dim=1)
        
        # Get action logits and value
        action_logits = self.action_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value
    


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Constants for DeepRMSA environment
        self.k_paths = 5
        self.max_path_length = 9
        self.edge_feature_dim = 16
        self.node_feature_dim = 1
        
        # Calculate feature dimensions
        self.num_edge_features = self.k_paths * self.max_path_length * self.edge_feature_dim
        self.num_node_features = self.k_paths * self.node_feature_dim
        self.num_service_info = 29  # Source + Destination + bit_rate
        
        print(f"Observation space shape: {observation_space.shape}")
        print(f"Edge features size: {self.num_edge_features}")
        print(f"Node features size: {self.num_node_features}")
        print(f"Service info size: {self.num_service_info}")
        
        # Initialize EGAT network
        self.egat_network = EGATNetwork(
            node_dim=self.node_feature_dim,
            edge_dim=self.edge_feature_dim,
            hidden_dim=64,
            num_layers=2,
            num_heads=8
        )
        
        # Initialize adjacency and mapping matrices
        self.register_buffer('AH', None)
        self.register_buffer('AE', None)
        self.register_buffer('ME', None)
        self.register_buffer('MH', None)

    def _build_adjacency_matrices(self, edge_features):
        """
        Build adjacency and mapping matrices for the topology
        Args:
            edge_features: [batch_size, k_paths, max_path_length, edge_dim]
        Returns:
            AH, AE, ME, MH matrices
        """
        batch_size = edge_features.size(0)
        device = edge_features.device
        
        # 1. Node Adjacency Matrix (AH)
        AH = torch.zeros(batch_size, self.k_paths, self.k_paths, device=device)
        
        # Connect nodes in same path and those sharing edges
        for i in range(self.k_paths):
            # Self-connections
            AH[:,i,i] = 1
            
            for j in range(i+1, self.k_paths):
                # Check if paths share any edges
                path_i = edge_features[:,i,:,:]
                path_j = edge_features[:,j,:,:]
                
                # If paths share any edges, connect their nodes
                shared = torch.any(torch.all(path_i.unsqueeze(2) == path_j.unsqueeze(1), dim=-1), dim=(-1,-2))
                AH[:,i,j] = shared
                AH[:,j,i] = shared  # Symmetric
        
        # 2. Edge Adjacency Matrix (AE)
        num_edges = self.k_paths * (self.max_path_length - 1)
        AE = torch.zeros(batch_size, num_edges, num_edges, device=device)
        
        # Connect edges that share a node in original topology
        edge_idx = 0
        for p in range(self.k_paths):
            for i in range(self.max_path_length-1):
                # Self connection
                AE[:,edge_idx,edge_idx] = 1
                
                # Connect with other edges
                other_idx = edge_idx + 1
                for q in range(p, self.k_paths):
                    start_j = 0 if q != p else i+1
                    for j in range(start_j, self.max_path_length-1):
                        # Check if edges share a node
                        edge1 = edge_features[:,p,i:i+2]
                        edge2 = edge_features[:,q,j:j+2]
                        
                        shares_node = torch.any(
                            torch.any(edge1.unsqueeze(2) == edge2.unsqueeze(1), dim=-1),
                            dim=(-1,-2)
                        )
                        AE[:,edge_idx,other_idx] = shares_node
                        AE[:,other_idx,edge_idx] = shares_node
                        other_idx += 1
                edge_idx += 1
        
        # 3. Edge Mapping Matrix (ME)
        ME = torch.zeros(batch_size, self.k_paths * self.k_paths, num_edges, device=device)
        
        edge_idx = 0
        for p in range(self.k_paths):
            for i in range(self.max_path_length-1):
                ME[:,p*self.k_paths:(p+1)*self.k_paths,edge_idx] = 1
                edge_idx += 1
        
        # 4. Node Mapping Matrix (MH)
        MH = torch.zeros(batch_size, num_edges, self.k_paths, device=device)
        
        edge_idx = 0
        for p in range(self.k_paths):
            for i in range(self.max_path_length-1):
                MH[:,edge_idx,p] = 1
                edge_idx += 1
        
        return AH, AE, ME, MH

    def _process_observation(self, obs):
        """
        Process observation tensor into network inputs
        Args:
            obs: Raw observation tensor [batch_size, total_features]
        Returns:
            node_features, edge_features, service_info
        """
        batch_size = obs.shape[0]
        
        # Extract features
        edge_features = obs[:, :self.num_edge_features].view(
            batch_size, 
            self.k_paths,
            self.max_path_length, 
            self.edge_feature_dim
        )
        
        node_features = obs[:, 
                          self.num_edge_features:self.num_edge_features+self.num_node_features
                         ].view(batch_size, self.k_paths, self.node_feature_dim)
        
        service_info = obs[:, -self.num_service_info:]
        
        # Build matrices if not already built
        if self.AH is None:
            self.AH, self.AE, self.ME, self.MH = self._build_adjacency_matrices(edge_features)
        
        return node_features, edge_features, service_info

    def forward(self, obs, deterministic=False):
        """
        Policy forward pass
        Args:
            obs: Observation tensor
            deterministic: Whether to sample or take best action
        Returns:
            actions, values, log_probs
        """
        # Process observation
        node_features, edge_features, service_info = self._process_observation(obs)
        
        # Forward through EGAT network
        action_logits, values = self.egat_network(
            node_features,
            edge_features,
            self.AH,
            self.AE,
            self.ME,
            self.MH,
            service_info
        )
        
        # Get action distribution
        distribution = self.get_distribution(action_logits)
        
        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for training
        Args:
            obs: Observation tensor
            actions: Action tensor
        Returns:
            values, log_probs, entropy
        """
        # Process observation
        node_features, edge_features, service_info = self._process_observation(obs)
        
        # Forward through EGAT network
        action_logits, values = self.egat_network(
            node_features,
            edge_features,
            self.AH,
            self.AE,
            self.ME,
            self.MH,
            service_info
        )
        
        distribution = self.get_distribution(action_logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_probs, entropy

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Get model's action prediction
        """
        observation = torch.as_tensor(observation).to(self.device)
        
        with torch.no_grad():
            actions, _, _ = self.forward(observation.unsqueeze(0), deterministic)
        
        return actions.cpu().numpy()

    def get_distribution(self, action_logits):
        """
        Convert logits to action distribution
        """
        action_probs = F.softmax(action_logits, dim=-1)
        return Categorical(probs=action_probs)
    


def main():
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

    # Monitor keywords
    monitor_info_keywords = (
        "service_blocking_rate",
        "episode_service_blocking_rate",
        "bit_rate_blocking_rate",
        "episode_bit_rate_blocking_rate"
    )

    # Create directories
    log_dir = "./tmp/deeprmsa-gat-a2c/"  # Changed directory name
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = gym.make('DeepRMSA-v0', **env_args)
    env = Monitor(env, log_dir + 'training', info_keywords=monitor_info_keywords)

    # Initialize A2C with custom GAT policy
    agent = A2C(
        CustomActorCriticPolicy,
        env,
        verbose=0,
        tensorboard_log="./tb/GAT-DeepRMSA-A2C/",  # Changed tensorboard directory
        learning_rate=7e-4,          # Default A2C learning rate
        n_steps=5,                   # A2C uses shorter trajectory steps
        gamma=0.99,                  # Discount factor
        gae_lambda=1.0,              # GAE parameter
        ent_coef=0.01,              # Entropy coefficient
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Gradient clipping
        rms_prop_eps=1e-5,          # RMSprop epsilon
        use_rms_prop=True,          # Use RMSprop optimizer
        normalize_advantage=True     # Normalize advantages
    )

    # Training callback with more frequent checks
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=100,  # More frequent checking for A2C
        log_dir=log_dir,
        verbose=1
    )

    # Train the agent
    agent.learn(
        total_timesteps=1000000,
        callback=callback
    )

if __name__ == "__main__":
    main()