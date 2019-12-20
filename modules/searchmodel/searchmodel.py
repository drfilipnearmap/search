import os
from datetime import datetime
from tqdm import tqdm_notebook
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfkm = tf.keras.models
tfpl = tfp.layers
tfd = tfp.distributions


def gpu_setup(gpu_number = 0, gpu_fraction = 0.45):
        
    """
    Set up tensorflow to run on a given GPU or CPU

    Arguments:
    gpu_number (int): -1 to run on the CPU or n to run on the nth GPU
    gpu_fraction (float): fraction of gpu memory to use, from 0 to 1

    Returns:
    None
    """

    from tensorflow.python import tf2
    if not tf2.enabled():
        print('enabling tf2')
        import tensorflow.compat.v2 as tf
        tf.enable_v2_behavior()
        assert tf2.enabled()

    import tensorflow.compat.v1 as tfc1
    
    GPU_DEV = gpu_number

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEV)
    if GPU_DEV >= 0:
        config = tfc1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        session = tfc1.Session(config=config)
    
    import tensorflow
    if tensorflow.test.gpu_device_name() != '/device:GPU:0':
        print('WARNING: GPU device not found.', tensorflow.test.gpu_device_name())
    else:
        print('SUCCESS: Found GPU: {}'.format(tensorflow.test.gpu_device_name()))
    
    tfk.backend.clear_session()


def res_block(x_in, num_filters, batchNorm=False, momentum=0.8, downsample=True):
    
    """
    Taken from Tristan at https://github.com/nearmap/notebooks/blob/master/personal/tristan/thesisSR/models/blocks.py, but with added downsampling
    Makes a Keras layer block which consists of 2 convolutional layers in a residual block and an additional convolutional layer to downsample, if necessary
    
    Arguments:
    x_in (Keras layer): input to the residual layer block, probably just the previous layer
    num_filters (int): number of convolutional filters, used in the Conv2D layers in the residual block
    batchNorm (boolean): whether to apply batch normalisation after the convolutional layers
    momentum (float): momentum value for the batch normalisation, if applicable
    downsample (boolean): whether to add a 3rd convolutional layer with strides = 2, which will effectively downscale the output
    
    Returns:
    x (Keras layer): the whole residual block as a single Keras layer. Used in the build_encoder() function
    """
    
    x = tfkl.Conv2D(num_filters, kernel_size=5, padding='same')(x_in)
    if batchNorm:
        x = tfkl.BatchNormalization(momentum=momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Conv2D(num_filters, kernel_size=5, padding='same')(x)
    if batchNorm:
        x = tfkl.BatchNormalization(momentum=momentum)(x)
    x = tfkl.Add()([x_in, x]) 
    
    if downsample==True:
        x = tfkl.LeakyReLU()(x)
        x = tfkl.Conv2D(num_filters, kernel_size=5, strides=2, padding='same')(x)
        x = tfkl.BatchNormalization(momentum = momentum)(x)
    return x


def res_block_transpose(x_in, num_filters, batchNorm=False, momentum=0.8, upsample=True):
    
    """
    Taken from Tristan at https://github.com/nearmap/notebooks/blob/master/personal/tristan/thesisSR/models/blocks.py, but with added upsampling. This is the same as res_block() but with transposed versions of the convolutional layers, used for upsampling instead of downsampling.
    Makes a Keras layer block which consists of 2 convolutional layers in a residual block and an additional convolutional layer to upsample, if necessary
    
    Arguments:
    x_in (Keras layer): input to the residual layer block, probably just the previous layer
    num_filters (int): number of convolutional filters, used in the Conv2D layers in the residual block
    batchNorm (boolean): whether to apply batch normalisation after the convolutional layers
    momentum (float): momentum value for the batch normalisation, if applicable
    upsample (boolean): whether to add a 3rd convolutional layer with strides = 2, which will effectively upsample the output
    
    Returns:
    x (Keras layer): the whole residual block as a single Keras layer. Used in the build_encoder() function
    """
    
    x = tfkl.Conv2DTranspose(num_filters, kernel_size=5, padding='same')(x_in)
    if batchNorm:
        x = tfkl.BatchNormalization(momentum=momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Conv2DTranspose(num_filters, kernel_size=5, padding='same')(x)
    if batchNorm:
        x = tfkl.BatchNormalization(momentum=momentum)(x)
    x = tfkl.Add()([x_in, x])
    x = tfkl.LeakyReLU()(x)
    
    if upsample==True:
        x = tfkl.Conv2DTranspose(num_filters, kernel_size=5, strides=2, padding='same')(x)
        x = tfkl.BatchNormalization(momentum = momentum)(x)
        x = tfkl.LeakyReLU()(x)
    return x

def build_encoder(prior, input_shape, encoded_size, base_depth, dropout_rate, momentum):
    """
    Returns an encoder Keras model for use in the build_vae() function
    
    Arguments:
    prior (tensorflow distribution): predefined in build_vae(), but this represents the distribution(s) with which to compare the latent dimensions, usually a set of normal distributions
    input_shape (tuple of ints): usually (128,128,1), this represents the size of the image being passed through the encoder
    encoded_size (int): number of latent dimensions in the variational auto encoder model
    base_depth (int): a base number of filters to be used in convolutional layers. Layers will use multiples of base_depth for their amount of filters
    dropout_rate (float): a value 0-1 representing the rate to apply to all tfkl.Dropout() layers
    momentum (float): a value 0-1 representing the rate to apply to all tfkl.BatchNormalization() layers
    
    Returns:
    Keras model representing the encoder. The encoder model has 4 outputs:
        - x_out: the latent space, as a vector of (<batch_size>,<encoded_size>) numbers
        - mu: a vector of <encoded_size>, representing the mean of each distribution in the latent space
        - log_var: a vector of <encoded_size>, representing the logarithmic variance of each distribution in the latent space
    """
    
    
    x_in = tfkl.Input(shape=input_shape)
    x = tfkl.Conv2D(2 * base_depth, 5, strides=2, padding='valid')(x_in)
    x = tfkl.BatchNormalization(momentum = momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(dropout_rate)(x)
    
    x = tfkl.Conv2D(4 * base_depth, 5, strides=2, padding='valid')(x)
    x = tfkl.BatchNormalization(momentum = momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(dropout_rate)(x)

    x = tfkl.Conv2D(4 * base_depth, 5, strides=2, padding='valid')(x)
    x = tfkl.BatchNormalization(momentum = momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(dropout_rate)(x)

    x = tfkl.Conv2D(2 * base_depth, 7, strides=2, padding='valid')(x)
    x = tfkl.BatchNormalization(momentum = momentum)(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(dropout_rate)(x)
    
    x = tfkl.Flatten()(x)

    mu      = tfkl.Dense(encoded_size, name='mu')(x)
    log_var = tfkl.Dense(encoded_size, name='log_var')(x)
    

#     # Reparameterization trick in order to get weights to backpropagate properly
#     # x_out = mu + exp(log_var) * epsilon, where epsilon is a random normal tensor.
    sigma         = tfkl.Lambda(lambda t: tfk.backend.exp(.5*t))(log_var)
    batch         = tfk.backend.shape(mu)[0]
    dim           = tfk.backend.shape(mu)[1]
    random_normal = tfk.backend.random_normal(mean = 0.0, 
                                              stddev = 1.0,
                                              shape=(batch, dim))
    eps           = tfkl.multiply([sigma, random_normal])
    x_out         = tfkl.add([mu, eps])

    return tfkm.Model(x_in, [x_out, mu, log_var], name='encoder')

def build_decoder(input_shape, encoded_size, base_depth, dropout_rate, momentum):
    
    """
    Returns a decoder Keras model for use in the build_vae() function
    
    Arguments:
    input_shape (tuple of ints): usually (128,128,1), this represents the size of the image being passed through the decoder
    encoded_size (int): number of latent dimensions in the variational auto encoder model
    base_depth (int): a base number of filters to be used in convolutional layers. Layers will use multiples of base_depth for their amount of filters
    dropout_rate (float): a value 0-1 representing the rate to apply to all tfkl.Dropout() layers
    momentum (float): a value 0-1 representing the rate to apply to all tfkl.BatchNormalization() layers
    
    Returns:
    Keras model representing the decoder. The decoder model has one output, which is the remade image. Note that every pixel is held as a Bernoulli distribution, rather than a single value, so that a logarithmic likelihood loss can be used.
    """
    
    x_in = tfkl.Input(shape=[encoded_size])
    x = tfkl.Reshape([1, 1, encoded_size])(x_in)

    x = tfkl.Dense(128*128)(x)

    x = tfkl.Flatten()(x)
    x = tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits)(x)

    return tfkm.Model(x_in, x, name='decoder')


def build_vae(encoded_size = 128):
    
    """
    Grab the encoder, decoder and constructed VAE for use in training or inference. This model shouldn't change once its been refined, so no arguments are provided.

    Arguments:
    encoded_size (int): how many distributions to store in the latent space

    Returns:
    (encoder, decoder, vae) tuple.
    Each of the three is a tensorflow.keras model, which can be trained and used for evaluating existing tiles with loaded or trained weights.
    """
    
    input_shape = [128,128,1]
    base_depth = 64
    
    dropout_rate = 0.2
    momentum = 0.99
    
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)

    encoder = build_encoder(prior, input_shape, encoded_size, base_depth, dropout_rate, momentum)
    decoder = build_decoder(input_shape, encoded_size, base_depth, dropout_rate, momentum)
    
    vae = tfk.Model(inputs=encoder.inputs,
                    outputs=decoder(encoder.outputs[0]))
    
    return (encoder, decoder, vae)


def build_encoder_resnet(prior, input_shape, encoded_size, base_depth, dropout_rate, momentum):
    """
    Returns an encoder Keras model for use in the build_vae() function. This structure differs to the one in build_encoder() as it uses residual blocks rather than just stacking conv layers together.
    
    Arguments:
    prior (tensorflow distribution): predefined in build_vae(), but this represents the distribution(s) with which to compare the latent dimensions, usually a set of normal distributions
    input_shape (tuple of ints): usually (128,128,1), this represents the size of the image being passed through the encoder
    encoded_size (int): number of latent dimensions in the variational auto encoder model
    base_depth (int): a base number of filters to be used in convolutional layers. Layers will use multiples of base_depth for their amount of filters
    dropout_rate (float): a value 0-1 representing the rate to apply to all tfkl.Dropout() layers
    momentum (float): a value 0-1 representing the rate to apply to all tfkl.BatchNormalization() layers
    
    Returns:
    Keras model representing the encoder. The encoder model has 3 outputs:
    x_out: the latent space, as a vector of (<batch_size>,<encoded_size>) numbers
    mu: a vector of <encoded_size>, representing the mean of each distribution in the latent space
    log_var: a vector of <encoded_size>, representing the logarithmic variance of each distribution in the latent space
    """
    x_in = tfkl.Input(shape=input_shape)
    x = res_block(x_in, 2 * base_depth, batchNorm=True, momentum=momentum, downsample=True)
    x = res_block(x, 2 * base_depth, batchNorm=True, momentum=momentum, downsample=True)
    x = res_block(x, 2 * base_depth, batchNorm=True, momentum=momentum, downsample=True)
    x = res_block(x, 2 * base_depth, batchNorm=True, momentum=momentum, downsample=True)
    
    x = tfkl.Flatten()(x)

    mu      = tfkl.Dense(encoded_size, name='mu')(x)
    log_var = tfkl.Dense(encoded_size, name='log_var')(x)
    
    sigma         = tfkl.Lambda(lambda t: tfk.backend.exp(.5*t))(log_var)
    batch         = tfk.backend.shape(mu)[0]
    dim           = tfk.backend.shape(mu)[1]
    random_normal = tfk.backend.random_normal(mean = 0.0, 
                                              stddev = 1.0,
                                              shape=(batch, dim))
    eps           = tfkl.multiply([sigma, random_normal])
    x_out         = tfkl.add([mu, eps])

    return tfkm.Model(x_in, [x_out, mu, log_var], name='encoder')


def build_vae_resnet(encoded_size = 128):
    
    """
    Grab the encoder, decoder and constructed VAE for use in training or inference. This model shouldn't change once its been refined, so no arguments are provided. This function differs to build_vae() as it uses the residual encoder model build_encoder_resnet() rather than the normal build_encoder(). The decoder is the same. 

    Arguments:
    encoded_size (int): how many distributions to store in the latent space

    Returns:
    (encoder, decoder, vae) tuple.
    Each of the three is a tensorflow.keras model, which can be trained and used for evaluating existing tiles with loaded or trained weights.

    """
    
    input_shape = [128,128,1]
    base_depth = 64
    
    dropout_rate = 0.2
    momentum = 0.99
    
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)

    encoder = build_encoder_resnet(prior, input_shape, encoded_size, base_depth, dropout_rate, momentum)
    decoder = build_decoder(input_shape, encoded_size, base_depth, dropout_rate, momentum)
    
    vae = tfk.Model(inputs=encoder.inputs,
                    outputs=decoder(encoder.outputs[0]))
    
    return (encoder, decoder, vae)


def kl_loss(mu, log_var):
    
    """
    Calculate the KL divergence between the latent space (given a mu and log var) and a normal distribution. This shouldn't be called manually, it's used by the train_step() and eval_step() functions for model training.
    
    Arguments:
    mu (float): Mean of the latent space.
    log_var (float): Logarithmic variance of the latent space.
    
    Returns:
    KL divergence (float).
    """
    
    kl_batch = - .5 * tfk.backend.sum(1 + log_var - tfk.backend.square(mu) - tfk.backend.exp(log_var), axis=-1)
    return tfk.backend.mean(kl_batch)

def nll_loss(true, pred):
    
    """
    Calculate the negative log likelihood between the remade image and original image. This shouldn't be called manually, it's used by the train_step() and eval_step() functions for model training.
    
    Arguments:
    true (numpy array): Original 128x128 prediction raster, with pixel values 0-1.
    pred (numpy array): Remade 128x128 prediction raster, where each pixel is a tfpl.IndependentBernoulli.
    
    Returns:
    Negative Log Likelihood (float).
    """
    
    return -pred.log_prob(true)

@tf.function
def train_step(p, beta, vae, encoder, decoder, optimiser):
    
    """
    Train the given model for one batch. This function shouldn't be called manually, it's used by train_model().
    
    Arguments:
    p (tensor): Batch of training images (128x128x1 prediction rasters) to use for training. This is of shape (batch size, 128, 128, 1) as per standard tf convention.
    beta (float): Value of hyperparameter beta, used to scale the KL divergence loss.
    vae (tfk model): Variational Auto Encoder that will be trained.
    encoder (tfk model): First half of the above vae, passed in for simplicity.
    decoder (tfk model): Second half of the above vae, passed in for simplicity.
    optimiser (tfk optimizer): Optimiser that will be updated by the training procedure.
    
    Returns:
    (kl, nll, diff) tuple
    
    kl (float): KL divergence using kl_loss().
    nll (float): Negative log-likelihood using nll_loss().
    diff (float): Mean absolute error between the original and remade images.
    
    """
    
    with tf.GradientTape() as tape:

        # These lines will augment the data to allow the model to learn rotation and translation invariance. 
        # I can't get it to train well :( Maybe you'll have better luck.
        
#         rotations = tf.random.uniform((p.shape[0],), minval = 0, maxval = 90)
#         p_rotated = tensorflow.contrib.image.rotate(p, rotations*math.pi/180.0)

#         translations = tf.random.uniform((p.shape[0],2), minval = -16, maxval = 16)
#         p_shifted = tensorflow.contrib.image.translate(p, translations)

        # Change this line to be 'encoder_output = encoder(p_shifted)' or 'encoder_output = encoder(p_rotated)' to use invariance
        encoder_output = encoder(p)
        vae_output = decoder(encoder_output[0])
        kl = kl_loss(encoder_output[1], encoder_output[2])

        nll = tf.math.reduce_mean(nll_loss(p, vae_output))
        diff = tf.math.reduce_sum(tf.abs(vae_output.mean() - p))

        loss = beta*kl + nll
        vae_gradients = tape.gradient(loss, vae.trainable_variables)
        optimiser.apply_gradients(zip(vae_gradients, vae.trainable_variables))

    return kl, nll, diff

@tf.function
def eval_step(p, encoder, decoder):
    
    """
    Evaluate the validation dataset during training, for one batch. This function shouldn't be called manually, it's used by train_model(). Similar to train_step(), except no gradients are calculated and there's no optimiser updates.
    
    Arguments: 
    p (tensor): Batch of validation images (128x128x1 prediction rasters) to use for evaluation. This is of shape (batch size, 128, 128, 1) as per standard tf convention.
    encoder (tfk model): First half of the full variational auto encoder model.
    decoder (tfk model): Second half of the full variational auto encoder model.
    
    Returns:
    (kl, nll, diff) tuple
    
    kl (float): KL divergence using kl_loss().
    nll (float): Negative log-likelihood using nll_loss().
    diff (float): Mean absolute error between the original and remade images.
    """
    
    encoder_output = encoder(p)
    vae_output = decoder(encoder_output[0])
    
    kl   = kl_loss(encoder_output[1], encoder_output[2])
    nll  = tf.math.reduce_mean(nll_loss(p, vae_output))
    diff = tf.math.reduce_sum(tf.abs(vae_output.mean() - p))
    
    return kl, nll, diff

def train_model(model, encoder, decoder, dataset_directory, epochs = 1000, learning_rate = 0.0001, batch_size = 20, beta = 0.1, run_name = 'vae', tensorboard_directory = "/mnt/DATA/data_filip/tensorboard_logs/"):
    
    """
    Given a tensorflow.keras model and dataset, train the model and log to tensorboard.
    
    Arguments:
    model (tfk model): The model to be trained. This should be a variational auto encoder, which takes in 128x128 images and remakes them by encoding them to a small latent space.
    encoder (tfk model): The first part of the above model, passed in for simplicity.
    decoder (tfk model): The second part of the above model, passed in for simplicity.
    dataset_directory (string): The base directory for the training data. This will be automatically split in to 90% training and 10% validation using tensorflow.keras.preprocessing.image.ImageDataGenerator(). This folder should have a 'train' (and 'test', for later if needed) subdirectory, which contains a single subdirectory named 'class' as the image data generator requires this folder structure. Therefore, the training dataset will actually reside in dataset_directory/train/class/.
    epochs (int): How many epochs to run training for.
    learning_rate (float): Learning rate for the optimiser.
    batch_size (int): How many images to use for training at once. 20 is a good number to start with.
    beta (float): Hyperparameter which scales the KL divergence in the training and evaluation loss.
    run_name (string): Name of the current training run for tensorboard.
    tensorboard_directory (string): Location in which to save the tensorboard logs.
    
    Returns:
    None. The provided model is trained in-place and can be used afterwards.
    """
    
    tfk.backend.clear_session()
    
    if dataset_directory[-1] == '/':
        dataset_directory = dataset_directory[:-1]
    if tensorboard_directory[-1] == '/':
        tensorboard_directory = tensorboard_directory[:-1]
    
    optimiser = tfk.optimizers.Adam(lr=learning_rate)
    log_dir = "{}/{}_{}".format(tensorboard_directory, run_name, datetime.now())
    file_writer = tf.summary.create_file_writer(log_dir)
    train_generator = tfk.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_batch = batch_size
    valid_batch = batch_size

    train_size = len(os.listdir('{}/train/class/'.format(dataset_directory)))
    valid_size = int(train_size)/10


    for e in tqdm_notebook(range(epochs)):

        train_flow      = train_generator.flow_from_directory(
                        directory   = '{}/train/'.format(dataset_directory), 
                        batch_size  = train_batch,
                        class_mode  = None,
                        color_mode  = 'grayscale',
                        target_size = (128, 128),
                        shuffle     = False,
                        subset      = "training")

        valid_flow      = train_generator.flow_from_directory(
                        directory   = '{}/train/'.format(dataset_directory), 
                        batch_size  = train_batch,
                        class_mode  = None,
                        color_mode  = 'grayscale',
                        target_size = (128, 128),
                        shuffle     = False,
                        subset      = "validation")

        train_loss  = 0
        train_diff  = 0
        train_kl    = 0
        train_nll   = 0

        valid_loss  = 0
        valid_kl    = 0
        valid_nll   = 0
        valid_diff  = 0

        i = 0
        train_im = 0
        test_im = 0


        for p in train_flow:

            # flow_from_directory() will loop indefinitely so we have to kill it manually
            if i == int(np.ceil(train_size/train_batch)):
                break
            i += 1

            kl, nll, diff = train_step(p, beta, model, encoder, decoder, optimiser)

            train_loss += (beta*kl + nll)
            train_diff += diff
            train_kl   += kl

        train_diff /= train_size
        train_loss /= train_size
        train_kl   /= train_size


        i = 0
        for p in valid_flow:

            # flow_from_directory() will loop indefinitely so we have to kill it manually
            if i == int(np.ceil(valid_size/valid_batch)):
                break
            i += 1

            kl, nll, diff = eval_step(p, encoder, decoder)

            valid_diff += diff
            valid_loss += (beta*kl + nll)
            valid_kl   += kl

        valid_diff  /= valid_size
        valid_loss  /= valid_size
        valid_kl    /= valid_size

        with file_writer.as_default():

            tf.summary.scalar('training_loss', train_loss, step = e)
            tf.summary.scalar('kl_train', train_kl, step = e)
            tf.summary.scalar('diff_train', train_diff, step = e)

            tf.summary.scalar('validation_loss', valid_loss, step = e)
            tf.summary.scalar('kl_valid', valid_kl, step = e)
            tf.summary.scalar('diff_valid', valid_diff, step = e)