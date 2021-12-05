import numpy as np
import tensorflow as tf

def get_map_trace(target_log_prob_fn, state, n_iter=1000, save_every=10, callbacks=[]):
    optimizer = tf.optimizers.Adam()
    @tf.function
    def minimize():
        optimizer.minimize(lambda: -target_log_prob_fn(*state), state)
    state_trace, cb_trace = [], [[] for _ in callbacks]
    for i in range(n_iter):
        if i % save_every == 0:
            state_trace.append(state)
            for trace, cb in zip(cb_trace, callbacks):
                trace.append(cb(state).numpy())
        minimize()
    return state_trace, cb_trace

def get_best_map_state(map_trace, map_log_probs):
    # map_log_probs[0/1]: train/test log probability
    test_set_max_log_prob_idx = np.argmax(map_log_probs[1])
    # Return MAP params that achieved highest test set likelihood.
    return map_trace[test_set_max_log_prob_idx]
    
def get_nodes_per_layer(n_features, net_taper=(1, 0.5, 0.2, 0.1)):
    nodes_per_layer = [int(n_features * x) for x in net_taper]
    # Ensure the last layer has two nodes so that output can be split into
    # predictive mean and learned loss attenuation (see eq. (7) of
    # https://arxiv.org/abs/1703.04977) which the network learns individually.
    nodes_per_layer.append(2)
    return nodes_per_layer