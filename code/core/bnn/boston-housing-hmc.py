
import tensorflow as tf
import tensorflow_probability as tfp

import bnn
from bnn import (
    get_random_initial_state,
    target_log_prob_fn_factory,
    tracer_factory,
)
from hmc import predict_from_chain, run_hmc
from map import get_best_map_state, get_map_trace, get_nodes_per_layer

with tf.device("/CPU:0"):
    # About the data: https://kaggle.com/c/boston-housing
    train, test = tf.keras.datasets.boston_housing.load_data()
    X_train, y_train, X_test, y_test = [arr.astype("float32") for arr in [*train, *test]]

    weight_prior = tfp.distributions.Normal(0, 0.2)
    bias_prior = tfp.distributions.Normal(0, 0.2)


    log_prob_tracers = (
        tracer_factory(X_train, y_train),
        tracer_factory(X_test, y_test),
    )
    n_features = X_train.shape[-1]
    random_initial_state = get_random_initial_state(
        weight_prior, bias_prior, get_nodes_per_layer(n_features)
    )
    trace, log_probs = get_map_trace(
        target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train),
        random_initial_state,
        n_iter=3000,
        callbacks=log_prob_tracers,
    )
    best_map_params = get_best_map_state(trace, log_probs)

    map_nn = bnn.build_net(best_map_params)
    map_y_pred, map_y_var = map_nn(X_test, training=False)

    bnn_log_prob_fn = target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train)
    _, samples, _, _ = run_hmc(bnn_log_prob_fn, current_state=best_map_params)
    hmc_y_pred, hmc_y_var = predict_from_chain(samples, X_test)
