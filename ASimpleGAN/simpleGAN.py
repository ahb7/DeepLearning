# This program trains a GAN using data with circular distribution
# Then, generates new synthetic data with the same properties 
# Executed with Python 3.7.10 and TensorFlow 2.1.0; On CPU, it takes 5mins
# Reference: Google TensorFlow GAN tutorials

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.datasets import make_circles

def generate_and_plot_data(model, test_input, train_data, epoch):
    plt.close()
    predictions = model(test_input, training=False)
    predictions = 100 * predictions; train_data = 100 * train_data
    generated = plt.scatter(predictions[:,0], predictions[:,1])
    real = plt.scatter(train_data[:,0], train_data[:,1])
    plt.legend((real, generated), ("Training Data","Generated Data"))
    plt.title('Data Samples Plot After {} Epoch'.format(epoch))
    plt.show(block=False)
    plt.pause(0.5)
    
# Create training data that is distributed as a circle
DATA_SIZE = 1000
x, y = make_circles(n_samples=DATA_SIZE, factor=0.999)
train_data = tf.cast(x, tf.float32)  # Already normalized in the range [-1, 1]

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_shape=(2,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(2))
    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_shape=(2,)))
    model.add(layers.LeakyReLU())          
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    return model

generator = generator_model()
discriminator = discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 20000; noise_dim = 2; num_examples_to_generate = 256
# Same seed is reused to generate data and visualize the progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(real_data):
    noise = tf.random.normal([DATA_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        train_step(dataset)
        if epoch%1000 == 0:
            print ('The epoch {} is done'.format(epoch+1))
            # Generate and plot the generated data as we train
            generate_and_plot_data(generator, seed, train_data, epoch+1)
    # Generate and plot the generated data at the end of the training
    generate_and_plot_data(generator, seed, train_data, epoch+1)
    print('The GAN training is done')
    plt.show()

# Start training, generating new data, and plotting them as we train
train(train_data, EPOCHS)
