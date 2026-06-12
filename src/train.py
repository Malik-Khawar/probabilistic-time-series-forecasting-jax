import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from src.losses import total_probabilistic_loss, pinball_loss, approximate_crps

def create_train_state(model, rng, learning_rate, input_shape):
    """Initializes the TrainState with optimizer and initial params."""
    dummy_input = jnp.zeros(input_shape)
    params = model.init(rng, dummy_input)['params']
    
    # Cosine decay schedule
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=2000,
        alpha=0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def make_epoch_step_fn(model, quantiles, horizon):
    """
    Creates a JIT-compiled function that runs one full epoch of training
    over pre-batched data using jax.lax.scan.
    """
    @jax.jit
    def train_epoch(state, batches):
        # batches is a tuple of (X_batches, y_batches)
        # X_batches shape: (num_batches, batch_size, lookback, num_features)
        # y_batches shape: (num_batches, batch_size, horizon + 2)
        X_batches, y_batches = batches
        
        def scan_body(carry_state, batch):
            x_b, y_b = batch
            y_target = y_b[:, :horizon]
            
            def loss_fn(params):
                y_pred = model.apply({'params': params}, x_b)
                loss = total_probabilistic_loss(y_target, y_pred, quantiles)
                return loss
                
            loss, grads = jax.value_and_grad(loss_fn)(carry_state.params)
            new_state = carry_state.apply_gradients(grads=grads)
            return new_state, loss
            
        final_state, losses = jax.lax.scan(scan_body, state, (X_batches, y_batches))
        return final_state, jnp.mean(losses)
        
    return train_epoch

def make_eval_step_fn(model, quantiles, horizon):
    """Creates a JIT-compiled evaluation function for validation/test data."""
    @jax.jit
    def eval_epoch(state, X, y):
        # X shape: (num_samples, lookback, num_features)
        # y shape: (num_samples, horizon + 2)
        y_target = y[:, :horizon]
        
        y_pred = model.apply({'params': state.params}, X)
        loss = pinball_loss(y_target, y_pred, quantiles)
        crps = approximate_crps(y_target, y_pred, quantiles)
        return loss, crps, y_pred
        
    return eval_epoch
