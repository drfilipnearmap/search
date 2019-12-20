import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Input, Lambda, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50

# fixed  model properties
OUTPUT_LAYER = 1
NUM_TOP_LAYERS = 4
FCN_8s_NUM_TOP_LAYERS = 8
FCN_4s_NUM_TOP_LAYERS = 12
NUM_BASE_LAYERS = 140


def make_dilated_fcn_resnet_16s(input_shape, nb_labels, weight_decay=0., batch_momentum=0.9, batch_norm=False,
                                depth=(2, 3, 5, 2), return_intermediate_features=False):
    """Create a resnet-50, resnet-101 or resnet-152 FCN
    Args:
        input_shape: Integer Tuple. Image (height, width,
            depth).
        nb_labels: Integer. The number of classes to predict.
        weight_decay:
        batch_momentum:
        batch_norm: Boolean. If True, include batch normalisation.
        depth: Integer Tuple. The number of identity blocks in
            each stage.
            (2, 3, 5, 2) [default] - Resnet 50
            (2, 3, 22, 2) - Resnet 101
            (2, 8, 5, 2) - Resnet 152
        return_intermediate_features: Boolean. If True, return links
            to intermediate stages in the model to be used by FPN
    Returns:
        A Keras Model object if return_intermediate_features=False.
        A tuple of tensors to be used by FPN:
            (input_tensor, stage_1, stage_2, stage_3, stage_4, stage_5)
    """

    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    bn_axis = 3

    if batch_norm is True:
        batch_norm = None

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(
        input_tensor)
    x = BatchNorm(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x, training=batch_norm)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    stage_1 = x

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1),
                   batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    for i in range(depth[0]):
        x = identity_block(3, [64, 64, 256], stage=2, block=chr(ord('b') + i), weight_decay=weight_decay,
                           batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    stage_2 = x

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_norm=batch_norm,
                   batch_momentum=batch_momentum)(x)
    for i in range(depth[1]):
        x = identity_block(3, [128, 128, 512], stage=3, block=chr(ord('b') + i), weight_decay=weight_decay,
                           batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    stage_3 = x

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_norm=batch_norm,
                   batch_momentum=batch_momentum)(x)
    for i in range(depth[2]):
        x = identity_block(3, [256, 256, 1024], stage=4, block=chr(ord('b') + i), weight_decay=weight_decay,
                           batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    stage_4 = x

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a0', weight_decay=weight_decay, atrous_rate=(2, 2),
                          batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    for i in range(depth[3]):
        x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b' + str(i), weight_decay=weight_decay,
                                  atrous_rate=(2, 2), batch_norm=batch_norm, batch_momentum=batch_momentum)(x)
    stage_5 = x

    x = Conv2D(nb_labels, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same',
               strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows, 'nb_cols': nb_cols}, name='resize_labels_8')(x)
    x = Activation('sigmoid')(x)
    model = Model(input_tensor, x)
    if return_intermediate_features:
        return [input_tensor, stage_1, stage_2, stage_3, stage_4, stage_5]
    else:
        return model


def proper_fcn(input_shape, nb_labels, model_type, weight_decay=0., batch_momentum=0.9, batch_norm=True):
    nb_rows, nb_cols, _ = input_shape
    if model_type.startswith('r_50'):
        model_depth = (2, 3, 5, 2)
    elif model_type.startswith('r_101'):
        model_depth = (2, 3, 22, 2)
    input_tensor, _, C2, C3, C4, C5 = make_dilated_fcn_resnet_16s(input_shape, nb_labels=nb_labels,
                                                                      weight_decay=weight_decay,
                                                                      batch_momentum=batch_momentum,
                                                                      batch_norm=batch_norm, depth=model_depth,
                                                                      return_intermediate_features=True)

    score = add_fcn_layers(C2, C3, C5, nb_rows, nb_cols, nb_labels, model_type)
    model = Model(input_tensor, score)
    if model_type.startswith('r_50_pretrained') or model_type.startswith('r_101_pretrained'):
        load_pretrained_weights(model)

    return model


def add_fcn_layers(C2, C3, C5, nb_rows, nb_cols, nb_labels, model_type):
    C5_score = Conv2D(nb_labels, (1, 1), strides=(1, 1), name='C5_score', padding="same")(C5)
    C5_upsamp = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows // 8, 'nb_cols': nb_cols // 8},
                       name='C5_upsamp')(C5_score)
    C3_stop_grad = Lambda(lambda x: K.stop_gradient(x))(C3)
    C3_score = Conv2D(nb_labels, (1, 1), strides=(1, 1), name='C3_score', padding="same")(C3_stop_grad)
    fuse_C3_C5 = Add(name='fuse_C3_C5')([C5_upsamp, C3_score])

    if model_type.endswith('fcn_8s'):
        fcn8s_upsamp = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows, 'nb_cols': nb_cols},
                              name='resize_labels_8s')(fuse_C3_C5)
        final_score = Activation('sigmoid', name='sigmoid_activation')(fcn8s_upsamp)
    elif model_type.endswith('fcn_4s'):
        C2_stop_grad = Lambda(lambda x: K.stop_gradient(x))(C2)
        C2_score = Conv2D(nb_labels, (1, 1), strides=(1, 1), name='C2_score', padding="same")(C2_stop_grad)
        combined_upsamp = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows // 4, 'nb_cols': nb_cols // 4},
                                 name='resize_8')(fuse_C3_C5)
        fuse_combined_C2 = Add(name='fuse_combined_C2')([combined_upsamp, C2_score])
        fcn4s_upsamp = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows, 'nb_cols': nb_cols},
                              name='resize_labels_4s')(fuse_combined_C2)
        final_score = Activation('sigmoid', name='sigmoid_activation')(fcn4s_upsamp)

    return final_score


def adapt_dilated_fcn_resnet_16s(base_model, input_shape, nb_labels, weight_decay=0., model_type=None):
    nb_rows, nb_cols, _ = input_shape
    if model_type.endswith('fcn_8s'):
        num_top_layers = FCN_8s_NUM_TOP_LAYERS
    elif model_type.endswith('fcn_4s'):
        num_top_layers = FCN_4s_NUM_TOP_LAYERS
    else:
        num_top_layers = NUM_TOP_LAYERS

    # get output of convolution layers
    convolutional_base = Model(base_model.inputs, base_model.layers[-num_top_layers].output)

    # when performing transfer learning, freeze the convolution base layers initially (warm-up stage) so the weights
    # aren't destroyed. After training the reinitialised layers, we can unfreeze the whole network and fine-tune
    for layer in convolutional_base.layers:
        layer.trainable = False
    if model_type.endswith('fcn_8s') or model_type.endswith('fcn_4s'):
        stage_2 = base_model.get_layer('res2c_out').output
        stage_3 = base_model.get_layer('res3d_out').output
        x = convolutional_base.layers[-OUTPUT_LAYER].output
        output = add_fcn_layers(stage_2, stage_3, x, nb_rows, nb_cols, nb_labels, model_type)
    else:
        # reinitialise task specific output layers
        x = convolutional_base.layers[-OUTPUT_LAYER].output
        x = Conv2D(nb_labels, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same',
                   strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
        x = Lambda(resize_bilinear, arguments={'nb_rows': nb_rows, 'nb_cols': nb_cols}, name='resize_labels_8')(x)
        output = Activation('sigmoid', name='sigmoid_activation')(x)

    model = Model(convolutional_base.inputs, output)
    return model


def pretrained_resnet(input_shape, nb_labels, model_type, batch_norm, weight_decay=0., batch_momentum=0.9):
    nb_rows, nb_cols, _ = input_shape
    if model_type.startswith('r_50'):
        model_depth = (2, 3, 5, 2)
    elif model_type.startswith('r_101'):
        model_depth = (2, 3, 22, 2)

    model = make_dilated_fcn_resnet_16s(input_shape, nb_labels, batch_norm=batch_norm, weight_decay=weight_decay,
                                        batch_momentum=batch_momentum,
                                        depth=model_depth, return_intermediate_features=False)
    load_pretrained_weights(model)

    return model

def load_pretrained_weights(model):
    model_path = get_imagenet_weights()
    load_weights(model, model_path, by_name=True)

    # when performing transfer learning, freeze the convolution base layers initially (warm-up stage) so the weights
    # aren't destroyed. After training the reinitialised layers, we can unfreeze the whole network and fine-tune
    final_fixed_layer = model.layers[NUM_BASE_LAYERS]
    if final_fixed_layer.name != 'res4f_out':
        raise NotImplementedError(f'Expected final non-trainable layer to be res4f_out, not {final_fixed_layer.name}')
    for layer in model.layers[:(NUM_BASE_LAYERS + 1)]:
        layer.trainable = False



def identity_block(kernel_size, filters, stage, block, batch_norm, weight_decay=0., batch_momentum=0.99):
    """Creates a residual identity block.
    The original help function from keras does not have weight
    regularizers.
    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        weight_decay:
        batch_momentum:
    """

    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        # In tensorflow it'll always be axis 3
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                   padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x, training=batch_norm)
        x = Add()([x, input_tensor])
        x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    return f


def conv_block(kernel_size, filters, stage, block, batch_norm, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        # In tensorflow it'll always be axis 3
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                   name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                   name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)
        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                          name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNorm(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut, training=batch_norm)
        x = Add()([x, shortcut])
        x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    return f


def atrous_identity_block(kernel_size, filters, stage, block, batch_norm, weight_decay=0., atrous_rate=(2, 2),
                          batch_momentum=0.99):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                   padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x, training=batch_norm)
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    return f


def atrous_conv_block(kernel_size, filters, stage, block, batch_norm, weight_decay=0., strides=(1, 1),
                      atrous_rate=(2, 2), batch_momentum=0.99):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                   name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                   name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x, training=batch_norm)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)
        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                          name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut, training=batch_norm)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    return f

# from tensorflow.compat.v1.image import resize_bilinear

def resize_bilinear(x, nb_rows, nb_cols):
    x = tf.compat.v1.image.resize_bilinear(x, [nb_rows, nb_cols], align_corners=True) # bilinear by default
    x.set_shape((None, nb_rows, nb_cols, None))
    return x


def get_imagenet_weights():
    from keras.utils.data_utils import get_file
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    return weights_path


def load_weights(model, filepath, by_name=False, exclude=None):
    import h5py
    from keras.engine import saving

    if exclude:
        by_name = True
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    keras_model = model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)
    if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()


class BatchNorm(BatchNormalization):
    """
    Extends the Keras BatchNormalization class to allow a central place to make changes if needed.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)
    
# Feature loss helper functions
def dilated_fcn_resnet_35(img_size=(256,256), 
                          path_to_weights='/mnt/DATA/data_nagita/models/Q4_34class/model_13.h5'): # activation_35
    return _dilated_fcn_resnet(img_size, path_to_weights) # Grabs the final activation size (None, 16, 16, 2048)

def _dilated_fcn_resnet(img_size, path_to_weights):
    res50 = make_dilated_fcn_resnet_16s((img_size[0], img_size[1], 3), 34)
    res50.load_weights(path_to_weights)
    return Model(res50.input, res50.output)