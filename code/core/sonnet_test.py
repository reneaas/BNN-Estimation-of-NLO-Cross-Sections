import tensorflow as tf
import sonnet as snt

if __name__ == "__main__":
    mlp = snt.Sequential([
        snt.Linear(10), tf.nn.relu,
        snt.Linear(1),
    ])

    n_train = 100
    x_train = tf.random.normal(shape=(n_train, 1))
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    y_train = f(x_train)

    with tf.GradientTape() as tape:
        y_pred = mlp(x_train)
        loss = tf.reduce_mean((y_pred - y_train) ** 2)
    grad = tape.gradient(loss, mlp.trainable_variables)

    new_vars = []
    print(mlp.trainable_variables)
    for w, dw in zip(mlp.trainable_variables, grad):
        w.assign_sub(w - dw)
    print(mlp.trainable_variables)




