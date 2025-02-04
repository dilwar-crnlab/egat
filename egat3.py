import tensorflow as tf


import numpy as np
import os
import pickle
import gym
from collections import deque
import time

import os
import pickle
import gym
import scipy

import numpy as np
from datetime import datetime
from tqdm import tqdm

class EGATLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=8, dropout_rate=0.2):
        super(EGATLayer, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Node feature transformation
        self.WH = self.add_weight(
            shape=(1, self.output_dim),  # For scalar betweenness
            initializer='glorot_uniform',
            name='node_transform'
        )

        # Edge feature transformation
        self.WE = self.add_weight(
            shape=(input_shape[1][-1], self.output_dim),  # For link features
            initializer='glorot_uniform',
            name='edge_transform'
        )

        # Node attention weights
        self.a = self.add_weight(
            shape=(2 * self.output_dim + self.output_dim, 1),
            initializer='glorot_uniform',
            name='node_attention'
        )

        # Edge attention weights
        self.b = self.add_weight(
            shape=(2 * self.output_dim + self.output_dim, 1),
            initializer='glorot_uniform',
            name='edge_attention'
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def node_attention(self, H, E, node_adj, training=False):
        """
        Compute attention for nodes in candidate paths.
        H: betweenness values [batch_size, num_nodes]
        E: edge features [batch_size, num_edges, edge_feat_dim]
        node_adj: adjacency matrix for nodes in paths [batch_size, num_nodes, num_nodes]
        """
        # Transform node features (betweenness)
        H_expanded = tf.expand_dims(H, -1)  # [batch, nodes, 1]
        H_transformed = H_expanded * self.WH  # [batch, nodes, output_dim]
        
        # Transform edge features
        E_transformed = tf.matmul(E, self.WE)  # [batch, edges, output_dim]
        
        # Compute attention scores
        N = tf.shape(H)[1]
        
        # Prepare node features for attention
        node_query = tf.expand_dims(H_transformed, 2)  # [batch, nodes, 1, dim]
        node_key = tf.expand_dims(H_transformed, 1)    # [batch, 1, nodes, dim]
        
        # Average edge features for each node pair based on adjacency
        # This assumes edge features correspond to node pairs indicated by node_adj
        edge_contrib = tf.reduce_mean(E_transformed, axis=1, keepdims=True)  # [batch, 1, output_dim]
        edge_contrib = tf.tile(edge_contrib, [1, N, N, 1])  # [batch, nodes, nodes, output_dim]
        
        # Concatenate features for attention
        attn_features = tf.concat([
            tf.tile(node_query, [1, 1, N, 1]),  # Source node features
            tf.tile(node_key, [1, N, 1, 1]),    # Target node features
            edge_contrib                         # Edge features
        ], axis=-1)
        
        # Compute attention scores
        e = tf.tanh(tf.einsum('bijf,fh->bij', attn_features, self.a))
        
        # Mask scores using node adjacency matrix
        e = tf.where(node_adj > 0, e, -1e9)
        
        # Apply softmax and dropout
        alpha = tf.nn.softmax(e, axis=-1)
        if training:
            alpha = self.dropout(alpha)
        
        # Compute new node features and edge-integrated features
        H_new = tf.einsum('bij,bjf->bif', alpha, H_transformed)
        Hm = tf.einsum('bij,bjf->bif', alpha, H_transformed * edge_contrib[:, :, :, 0])
        
        return H_new, Hm, alpha

    def edge_attention(self, H, E, edge_adj, training=False):
        """
        Compute attention for edges in candidate paths.
        H: betweenness values [batch_size, num_nodes]
        E: edge features [batch_size, num_edges, edge_feat_dim]
        edge_adj: adjacency matrix for edges in paths [batch_size, num_edges, num_edges]
        """
        # Transform edge features
        E_transformed = tf.matmul(E, self.WE)  # [batch, edges, output_dim]
        
        # Average node features for each edge
        H_avg = tf.reduce_mean(tf.expand_dims(H, -1), axis=1, keepdims=True)  # [batch, 1, 1]
        H_avg = tf.tile(H_avg, [1, tf.shape(E)[1], 1])  # [batch, edges, 1]
        
        M = tf.shape(E)[1]
        
        # Prepare edge features for attention
        edge_query = tf.expand_dims(E_transformed, 2)  # [batch, edges, 1, dim]
        edge_key = tf.expand_dims(E_transformed, 1)    # [batch, 1, edges, dim]
        
        # Prepare node contribution
        node_contrib = tf.tile(tf.expand_dims(H_avg, 2), [1, 1, M, 1])  # [batch, edges, edges, 1]
        
        # Concatenate features for attention
        attn_features = tf.concat([
            tf.tile(edge_query, [1, 1, M, 1]),  # Source edge features
            tf.tile(edge_key, [1, M, 1, 1]),    # Target edge features
            node_contrib                         # Node features
        ], axis=-1)
        
        # Compute attention scores
        e = tf.tanh(tf.einsum('bijf,fh->bij', attn_features, self.b))
        
        # Mask scores using edge adjacency matrix
        e = tf.where(edge_adj > 0, e, -1e9)
        
        # Apply softmax and dropout
        beta = tf.nn.softmax(e, axis=-1)
        if training:
            beta = self.dropout(beta)
        
        # Compute new edge features
        E_new = tf.einsum('bij,bjf->bif', beta, E_transformed)
        
        return E_new, beta

    def call(self, inputs, training=False):
        betweenness, link_features, node_adj, edge_adj = inputs
        # print("Betweenness in Call", betweenness)
        # print("Link features", link_features)
        # print("Node adj", node_adj),
    
        
        # Compute node attention
        H_new, Hm, alpha = self.node_attention(betweenness, link_features, node_adj, training)
        
        # Compute edge attention
        E_new, beta = self.edge_attention(betweenness, link_features, edge_adj, training)
        
        return H_new, E_new, Hm, alpha, beta

class DeepRMSAFeatureExtractor(tf.keras.Model):
    def __init__(self, num_original_nodes, num_edges, k_paths, num_bands, P, hidden_size):
        super(DeepRMSAFeatureExtractor, self).__init__()
        self.num_original_nodes = num_original_nodes
        self.num_edges = num_edges
        self.k_paths = k_paths
        self.num_bands = num_bands
        self.hidden_size = hidden_size
        self.P = P
        
        # Single EGAT layer
        self.egat = EGATLayer(hidden_size)
        
        # Dense layers
        self.dense_layers = [
            tf.keras.layers.Dense(hidden_size, activation='relu')
            for _ in range(5)
        ]

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        idx = 0
        print("Batch size", batch_size, idx)
        
        # Extract components
        source_dest_size = 2 * self.num_original_nodes
        source_dest = inputs[:, idx:idx + source_dest_size]
        idx += source_dest_size
        
        slots_path_size = self.k_paths
        slots_per_path = inputs[:, idx:idx + slots_path_size]
        idx += slots_path_size
        
        spectrum_size = self.k_paths * 6 * self.num_bands
        spectrum = inputs[:, idx:idx + spectrum_size]
        spectrum = tf.reshape(spectrum, [batch_size, self.k_paths, self.num_bands, 6])
        idx += spectrum_size
        
        # Link features for edges in candidate paths
        features_per_band = 8
        general_link_features = 1
        features_per_link = general_link_features + (features_per_band * self.num_bands)
        link_features_size = self.num_edges * features_per_link
        link_features = inputs[:, idx:idx + link_features_size]
        link_features = tf.reshape(link_features, [batch_size, self.num_edges, -1])
        idx += link_features_size
        
        # Node features (betweenness)
        betweenness_size = self.num_original_nodes
        betweenness = inputs[:, idx:idx + betweenness_size]
        idx += betweenness_size
        
        # Adjacency matrices for nodes and edges in candidate paths
        node_adj_size = self.num_original_nodes * self.num_original_nodes
        node_adj = inputs[:, idx:idx + node_adj_size]
        node_adj = tf.reshape(node_adj, [batch_size, self.num_original_nodes, self.num_original_nodes])
        idx += node_adj_size
        
        edge_adj_size = self.num_edges * self.num_edges
        edge_adj = inputs[:, idx:idx + edge_adj_size]
        edge_adj = tf.reshape(edge_adj, [batch_size, self.num_edges, self.num_edges])
        
        # EGAT processing
        H_new, E_new, Hm, alpha, beta = self.egat([betweenness, link_features, node_adj, edge_adj],training=training)
        
        # Concatenate features
        c_band = spectrum[:, :, 0, :]
        l_band = spectrum[:, :, 1, :]
        
        combined = tf.concat([
            Hm,  # Edge-integrated node features
            source_dest,
            slots_per_path,
            tf.reshape(c_band, [batch_size, self.k_paths * 6]),
            tf.reshape(l_band, [batch_size, self.k_paths * 6]),
            tf.reshape(alpha, [batch_size, -1])  # Node attention weights
        ], axis=1)
        
        # Final dense processing
        x = combined
        for dense in self.dense_layers:
            x = dense(x)
            
        return x


class AC_Net(tf.keras.Model):
    def __init__(self, scope, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar,
                 num_original_nodes, num_edges, k_paths, num_bands, P):
        super(AC_Net, self).__init__()
        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar
        
        # Feature extraction
        self.feature_extractor = DeepRMSAFeatureExtractor(
            num_original_nodes=num_original_nodes,
            num_edges=num_edges,
            k_paths=k_paths,
            num_bands=num_bands,
            P=P,
            hidden_size=layer_size
        )
        
        # Policy network
        self.policy_layer = tf.keras.layers.Dense(
            n_actions,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
            kernel_initializer=self.normalized_columns_initializer(0.01),
            use_bias=False,
            name='policy'
        )
        
        # Value network
        self.value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
            kernel_initializer=self.normalized_columns_initializer(1.0),
            use_bias=False,
            name='value'
        )

    def call(self, inputs, training=False):
        print("Inputs", inputs)
        features = self.feature_extractor(inputs, training=training)
        policy = self.policy_layer(features)
        value = self.value_layer(features)
        return policy, value

    @tf.function
    def compute_losses(self, policy, value, actions, target_v, advantages):
        actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)
        
        # Policy loss
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-6))
        policy_loss = -tf.reduce_sum(tf.math.log(responsible_outputs + 1e-6) * advantages)
        total_policy_loss = policy_loss - 0.01 * entropy
        
        # Value loss
        value_loss = tf.reduce_sum(tf.square(target_v - tf.reshape(value, [-1])))
        
        return total_policy_loss, value_loss, entropy

    @tf.function
    def train_step(self, inputs, actions, target_v, advantages):
        with tf.GradientTape(persistent=True) as tape:
            policy, value = self(inputs, training=True)
            policy_loss, value_loss, entropy = self.compute_losses(
                policy, value, actions, target_v, advantages
            )
        
        # Get gradients
        policy_grads = tape.gradient(policy_loss, self.policy_layer.trainable_variables)
        value_grads = tape.gradient(value_loss, self.value_layer.trainable_variables)
        
        # Clip gradients
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 40.0)
        value_grads, _ = tf.clip_by_global_norm(value_grads, 40.0)
        
        return policy_loss, value_loss, entropy, policy_grads, value_grads

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

class DeepRMSA_A3C:
    def __init__(self, env, num_layers=4, layer_size=128, learning_rate=1e-4, regu_scalar=1e-3):
        self.env = env

        # Calculate link features size
        features_per_band = 8  # 8 features per band
        general_link_features = 1  # spans_score
        features_per_link = general_link_features + (features_per_band * env.num_bands)
        link_features_size = env.topology.number_of_edges() * features_per_link
        # Calculate total x_dim
        self.x_dim = (
            2 * env.topology.number_of_nodes() +    # source_dest (2 one-hot vectors)
            env.topology.number_of_nodes() +        # node betweenness
            env.k_paths +                          # slots_per_path
            (env.k_paths * 6 * env.num_bands) +    # spectrum distribution 
            link_features_size +                    # link features
            (env.topology.number_of_nodes() * env.topology.number_of_nodes()) +  # node adjacency
            (env.topology.number_of_edges() * env.topology.number_of_edges())    # edge adjacency
        )
        print("X_dim ", self.x_dim)

        print("Component sizes:")
        print(f"Source-dest: {2 * env.topology.number_of_nodes()}")
        print(f"Betweenness: {env.topology.number_of_nodes()}")
        print(f"Slots per path: {env.k_paths}")
        print(f"Spectrum features: {env.k_paths * 6 * env.num_bands}")
        print(f"Link features: {link_features_size}")
        print(f"Node adjacency: {env.topology.number_of_nodes() * env.topology.number_of_nodes()}")
        print(f"Edge adjacency: {env.topology.number_of_edges() * env.topology.number_of_edges()}")
        print(f"Total x_dim: {self.x_dim}")

        # Find longest path for GCN steps
        k_shortest_paths = env.topology.graph["ksp"]
        self.P = self.find_longest_path_by_hops(k_shortest_paths)
        
        # Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Networks
        self.global_network = AC_Net(
            scope='global',
            x_dim_p=self.x_dim,
            x_dim_v=self.x_dim,
            n_actions=env.action_space.n,
            num_layers=num_layers,
            layer_size=layer_size,
            regu_scalar=regu_scalar,
            num_original_nodes=env.topology.number_of_nodes(),
            num_edges=env.topology.number_of_edges(),
            k_paths=env.k_paths,
            num_bands=env.num_bands,
            P=self.P
        )
        
        self.local_network = AC_Net(
            scope='worker',
            x_dim_p=self.x_dim,
            x_dim_v=self.x_dim,
            n_actions=env.action_space.n,
            num_layers=num_layers,
            layer_size=layer_size,
            regu_scalar=regu_scalar,
            num_original_nodes=env.topology.number_of_nodes(),
            num_edges=env.topology.number_of_edges(),
            k_paths=env.k_paths,
            num_bands=env.num_bands,
            P=self.P
        )
        
        # Initialize networks
        dummy_state = tf.zeros([1, self.x_dim])
        self.global_network(dummy_state)
        self.local_network(dummy_state)
        
        # Copy weights from global to local network
        self.sync_networks()
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_blocking_rates = []

    def find_longest_path_by_hops(self, k_shortest_paths):
        max_hops = 0
        for (src, dst), paths in k_shortest_paths.items():
            for path in paths:
                num_hops = len(path.node_list) - 1
                if num_hops > max_hops:
                    max_hops = num_hops
        return max_hops

    @tf.function
    def choose_action(self, state, deterministic=False):
        policy, _ = self.local_network(tf.expand_dims(state, 0))
        policy = policy[0]
        
        if deterministic:
            return tf.argmax(policy)
        else:
            return tf.random.categorical(tf.math.log(policy + 1e-10)[None, :], 1)[0, 0]

    def sync_networks(self):
        """Copy weights from global to local network."""
        for local_var, global_var in zip(
            self.local_network.trainable_variables,
            self.global_network.trainable_variables
        ):
            local_var.assign(global_var)

    @tf.function
    def update_networks(self, states, actions, returns, advantages):
        # Update networks using gradient tape
        policy_loss, value_loss, entropy, policy_grads, value_grads = \
            self.local_network.train_step(states, actions, returns, advantages)
            
        # Apply gradients to global network
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.global_network.policy_layer.trainable_variables)
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.global_network.value_layer.trainable_variables)
        )
        
        # Sync networks
        self.sync_networks()
        
        return policy_loss, value_loss, entropy

    def train(self, total_timesteps, gamma=0.99, bootstrap_value=True, n_steps=5):
        episode_count = 0
        total_steps = 0
        current_service_count = 0  # Count services within an episode
        start_time = time.time()
        
        # Initialize buffers for n_steps
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        print("Initial state after reset:", state[:28])
        episode_reward = 0

        # Create progress bar for episode length
        pbar = tqdm(total=self.env.episode_length, desc="Episode Progress")
        
        while total_steps < total_timesteps:
            print("State before choose_action:", state[:28])
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action.numpy())
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            episode_reward += reward
            total_steps += 1
            current_service_count += 1
            state = next_state

            # Update progress bar
            pbar.update(1)
            
            # Update network every n service requests or at episode end
            if current_service_count % n_steps == 0 or done:
                # Get bootstrap value
                if bootstrap_value:
                    _, value = self.local_network(tf.expand_dims(state, 0))
                    bootstrap_value = value.numpy()[0, 0]
                else:
                    bootstrap_value = 0
                    
                # Convert to tensors
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards_array = np.array(rewards)
                
                # Compute returns and advantages
                rewards_plus = np.append(rewards_array, bootstrap_value)
                discounted_returns = tf.convert_to_tensor(
                    self.discount(rewards_plus, gamma)[:-1], 
                    dtype=tf.float32
                )
                
                # Get values
                _, values = self.local_network(states_tensor)
                values = tf.squeeze(values)
                
                # Compute advantages
                advantages = discounted_returns - values
                
                # Update networks
                policy_loss, value_loss, entropy = self.update_networks(
                    states_tensor, actions_tensor, discounted_returns, advantages
                )
                
                # Clear buffers
                states = []
                actions = []
                rewards = []
            
            # Handle episode completion (100 service requests)
            if done:
                episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_blocking_rates.append(
                    info.get('episode_service_blocking_rate', 0)
                )
                
                # Reset episode stats
                state = self.env.reset()
                episode_reward = 0
                current_service_count = 0

                # Reset progress bar for next episode
                pbar.reset()
                
                # Log progress
                #if episode_count % 1 == 0:  # Log every episode
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_blocking = np.mean(self.episode_blocking_rates[-10:])
                fps = int(total_steps / (time.time() - start_time))
                    # print(f"Episode: {episode_count}")
                    # print(f"Mean Reward: {mean_reward:.2f}")
                    # print(f"Mean Blocking Rate: {mean_blocking:.4f}")
                    # print(f"Policy Loss: {policy_loss:.4f}")
                    # print(f"Value Loss: {value_loss:.4f}")
                    # print(f"Entropy: {entropy:.4f}")
                    # print(f"FPS: {fps}\n")

                pbar.set_postfix({
                        'episode': episode_count,
                        'mean_reward': f'{mean_reward:.2f}',
                        'mean_blocking': f'{mean_blocking:.4f}',
                        # 'fps': fps,
                        # 'policy_loss': f'{policy_loss:.4f}',
                        # 'value_loss': f'{value_loss:.4f}'
                    })
    
        # Close progress bar
        pbar.close()
        
        return self

    def log_progress(self, episode_count, total_steps, total_timesteps, 
                    start_time, policy_loss, value_loss, entropy):
        mean_reward = np.mean(self.episode_rewards[-100:])
        mean_length = np.mean(self.episode_lengths[-100:])
        mean_blocking = np.mean(self.episode_blocking_rates[-100:])
        fps = int(total_steps / (time.time() - start_time))
        
        print(f"Episode: {episode_count}")
        print(f"Steps: {total_steps}/{total_timesteps}")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Mean Length: {mean_length:.2f}")
        print(f"Mean Blocking Rate: {mean_blocking:.4f}")
        print(f"FPS: {fps}")
        print(f"Policy Loss: {policy_loss:.4f}")
        print(f"Value Loss: {value_loss:.4f}")
        print(f"Entropy: {entropy:.4f}\n")

    def discount(self, x, gamma):
        """Compute discounted returns."""
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]





class SaveOnBestTrainingCallback:
    def __init__(self, check_freq, log_dir, verbose=1):
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.verbose = verbose
        self.best_mean_reward = -np.inf
        
    def on_step(self, locals, globals):
        if len(locals['self'].episode_rewards) < self.check_freq:
            return True
            
        mean_reward = np.mean(locals['self'].episode_rewards[-self.check_freq:])
        
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"Saving new best model with mean reward: {mean_reward:.2f}")
            locals['self'].save(os.path.join(self.log_dir, 'best_model'))
            
        return True

def main():
    # Set random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Set up TensorFlow GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Load topology
    topology_name = 'nsfnet_chen'
    k_paths = 5
    with open(f'../topologies/{topology_name}_{k_paths}-paths_new.h5', 'rb') as f:
        topology = pickle.load(f)

    # Environment setup
    env_args = dict(
        num_bands=2,
        topology=topology,
        seed=10,
        allow_rejection=False,
        j=1,
        mean_service_holding_time=15.0,
        mean_service_inter_arrival_time=0.1,
        k_paths=k_paths,
        episode_length=1000,
        node_request_probabilities=None
    )

    # Create directories with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"./logs/deeprmsa-a3c/{current_time}"
    model_dir = f"./models/deeprmsa-a3c/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Set up TensorBoard
    #summary_writer = tf.summary.create_file_writer(log_dir)

    # Create and wrap environment
    env = gym.make('DeepRMSA-v0', **env_args)
    
    # Create A3C model
    model = DeepRMSA_A3C(
        env=env,
        num_layers=4,
        layer_size=128,
        learning_rate=1e-4,
        regu_scalar=1e-3
    )

    # Create callback
    callback = SaveOnBestTrainingCallback(
        check_freq=100,
        log_dir=model_dir,
        verbose=1,
        #summary_writer=summary_writer
    )

    # Training hyperparameters
    training_params = {
        'total_timesteps': 1000000,
        'gamma': 0.95,
        'bootstrap_value': True
    }

    # Save training parameters
    with open(os.path.join(log_dir, 'training_params.txt'), 'w') as f:
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")

    # Log configuration
    print("Starting training...")
    print(f"Logging to {log_dir}")
    print(f"Models will be saved to {model_dir}")
    print("\nTraining parameters:")
    for key, value in training_params.items():
        print(f"{key}: {value}")
    print("\nEnvironment parameters:")
    for key, value in env_args.items():
        print(f"{key}: {value}")
    print("\nModel parameters:")
    print(f"Number of layers: {model.local_network.num_layers}")
    print(f"Layer size: {model.local_network.layer_size}")
    print(f"Learning rate: {model.policy_optimizer.learning_rate.numpy()}")
    print(f"Regularization scalar: {model.local_network.regu_scalar}")

    try:
        # Train the model
        model.train(
            total_timesteps=training_params['total_timesteps'],
            gamma=training_params['gamma'],
            bootstrap_value=training_params['bootstrap_value'],
            #callback=callback
            n_steps=1
        )
        
        # Save final model
        final_model_path = "./models/deeprmsa-a3c/20250127-205303/final_model.weights.h5"
        model.global_network.save_weights(final_model_path)
        print(f"\nTraining completed. Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupted model
        interrupted_model_path = os.path.join(model_dir, 'interrupted_model')
        model.global_network.save_weights(interrupted_model_path)
        print(f"Interrupted model saved to {interrupted_model_path}")
        
    finally:
        # Save training curves
        training_data = {
            'rewards': model.episode_rewards,
            'lengths': model.episode_lengths,
            'blocking_rates': model.episode_blocking_rates
        }
        np.save(os.path.join(log_dir, 'training_data.npy'), training_data)
        
        # Close environment
        env.close()
        
        # Final TensorBoard flush
        #summary_writer.flush()

if __name__ == "__main__":
    main()

