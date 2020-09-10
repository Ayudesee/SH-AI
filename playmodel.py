import tensorflow as tf

filepath = 'models/model-9-10-21-36'

model = tf.keras.models.load_model(filepath=filepath)
