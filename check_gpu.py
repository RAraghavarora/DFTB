import tensorflow as tf
devices = tf.config.experimental.list_physical_devices("GPU")
print(devices)