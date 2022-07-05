# Defines a set of trace functions used with a finite
# set of transition kernels provided by TensorFlow Probability's
# MCMC module.


def trace_fn_hmc(_, pkr):
    return {"is_accepted": pkr.is_accepted}


def trace_fn_nuts(_, pkr):
    return {
        "target_log_prob": pkr.target_log_prob,
        "leapfrogs_taken": pkr.leapfrogs_taken,
        "has_divergence": pkr.has_divergence,
        "energy": pkr.energy,
        "log_accept_ratio": pkr.log_accept_ratio,
        "is_accepted": pkr.is_accepted,
    }


def trace_fn_adaptive_hmc(_, pkr):
    return {
        "is_accepted": pkr.inner_results.is_accepted,
        "step_size": pkr.inner_results.accepted_results.step_size,
        "target_accept_prob": pkr.target_accept_prob,
    }


def trace_fn_adaptive_nuts(_, pkr):
    return {
        "target_log_prob": pkr.inner_results.target_log_prob,
        "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
        "has_divergence": pkr.inner_results.has_divergence,
        "energy": pkr.inner_results.energy,
        "log_accept_ratio": pkr.inner_results.log_accept_ratio,
        "is_accepted": pkr.inner_results.is_accepted,
        "step_size": pkr.inner_results.step_size,
        "target_accept_prob": pkr.target_accept_prob,
    }