import os

from keras.utils import conv_utils
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import skimage
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAvgPool2D, Concatenate
# from keras.utils import conv_utils
from skimage.io import imread
import numpy as np
from tensorflow import keras
from keras import layers
from keras.regularizers import Regularizer
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

target_folder = '/cluster/simulations_400_RGB/'

# LOAD IMAGES
X = []
Y = []
for i in range(1, 797, 1):
    I = imread(target_folder + 'cell_edge_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(0)
    I = imread(target_folder + 'extranuclear_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(1)
    I = imread(target_folder + 'foci_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(2)
    I = imread(target_folder + 'intranuclear_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(3)
    I = imread(target_folder + 'nuclear_edge_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(4)
    I = imread(target_folder + 'pericellular_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(5)
    I = imread(target_folder + 'perinuclear_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(6)
    I = imread(target_folder + 'random_' + str(i) + '_400' + '.ppm')[:, :, 0]
    X.append(I)
    Y.append(7)
    # print(I.sum())

X = np.array(X)
print("intial shape", X.shape)
X = np.expand_dims(X, axis=-1)
print(X.shape)
Y = np.array(Y)
Y = np.expand_dims(Y, axis=-1)
print(Y.shape)

# Scale images to the [0, 1] range
X = X.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
print("X shape:", X.shape)
print(X.shape[0], "train samples")

Xtrain = X[0:5400]
Ytrain = Y[0:5400]
Xval = X[5400:]
Yval = Y[5400:]
print(Xtrain.shape)
print(Ytrain.shape)
print(Yval.shape)

target_folder_2 = '/cluster/mask_400/'
# LOAD IMAGES
X_2 = []
Y_2 = []
for i in range(1, 797, 1):
    I = imread(target_folder_2 + 'cell_edge_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(0)
    I = imread(target_folder_2 + 'extranuclear_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(1)
    I = imread(target_folder_2 + 'foci_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(2)
    I = imread(target_folder_2 + 'intranuclear_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(3)
    I = imread(target_folder_2 + 'nuclear_edge_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(4)
    I = imread(target_folder_2 + 'pericellular_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(5)
    I = imread(target_folder_2 + 'perinuclear_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(6)
    I = imread(target_folder_2 + 'random_sub_mask_' + str(i) + '_400' + '.png')
    X_2.append(I)
    Y_2.append(7)
    # print(I.sum())

X_2=np.array(X_2)
Y_2=np.array(Y_2)

X2train=X_2[0:5400]
Y2train=Y_2[0:5400]
X2val=X_2[5400:]
Y2val=Y_2[5400:]

# LOAD IMAGES
X_3 = []
Y_3 = []
for i in range(1, 797, 1):
    I = imread(target_folder_2 + 'cell_edge_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(0)
    I = imread(target_folder_2 + 'extranuclear_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(1)
    I = imread(target_folder_2 + 'foci_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(2)
    I = imread(target_folder_2 + 'intranuclear_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(3)
    I = imread(target_folder_2 + 'nuclear_edge_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(4)
    I = imread(target_folder_2 + 'pericellular_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(5)
    I = imread(target_folder_2 + 'perinuclear_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(6)
    I = imread(target_folder_2 + 'random_nuc_mask_' + str(i) + '_400' + '.png')
    X_3.append(I)
    Y_3.append(7)
    # print(I.sum())

X_3=np.array(X_3)
Y_3=np.array(Y_3)

X3train=X_3[0:5400]
Y3train=Y_3[0:5400]
X3val=X_3[5400:]
Y3val=Y_3[5400:]


# LOAD IMAGES
X_4 = []
Y_4 = []
for i in range(1, 797, 1):
    I = imread(target_folder_2 + 'cell_edge_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(0)
    I = imread(target_folder_2 + 'extranuclear_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(1)
    I = imread(target_folder_2 + 'foci_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(2)
    I = imread(target_folder_2 + 'intranuclear_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(3)
    I = imread(target_folder_2 + 'nuclear_edge_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(4)
    I = imread(target_folder_2 + 'pericellular_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(5)
    I = imread(target_folder_2 + 'perinuclear_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(6)
    I = imread(target_folder_2 + 'random_cyt_mask_' + str(i) + '_400' + '.png')
    X_4.append(I)
    Y_4.append(7)
    # print(I.sum())

X_4=np.array(X_4)
Y_4=np.array(Y_4)

X4train=X_4[0:5400]
Y4train=Y_4[0:5400]
X4val=X_4[5400:]
Y4val=Y_4[5400:]


# LOAD IMAGES
X_5 = []
Y_5 = []
for i in range(1, 797, 1):
    I = imread(target_folder_2 + 'cell_edge_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(0)
    I = imread(target_folder_2 + 'extranuclear_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(1)
    I = imread(target_folder_2 + 'foci_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(2)
    I = imread(target_folder_2 + 'intranuclear_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(3)
    I = imread(target_folder_2 + 'nuclear_edge_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(4)
    I = imread(target_folder_2 + 'pericellular_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(5)
    I = imread(target_folder_2 + 'perinuclear_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(6)
    I = imread(target_folder_2 + 'random_cyt_reverse_mask_' + str(i) + '_400' + '.png')
    X_5.append(I)
    Y_5.append(7)
    # print(I.sum())

X_5=np.array(X_5)
Y_5=np.array(Y_5)

X5train=X_5[0:5400]
Y5train=Y_5[0:5400]
X5val=X_5[5400:]
Y5val=Y_5[5400:]



MIN_LATT = -1


class L1L2Lattice(Regularizer):
    """Regularizer for L1 and L2 regularization in a lattice.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = tf.cast(l1, dtype=tf.float32)
        self.l2 = tf.cast(l2, dtype=tf.float32)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.math.abs(x - MIN_LATT) * tf.math.abs(x))
        if self.l2:
            regularization += self.l2 * tf.reduce_sum(tf.math.square(x - MIN_LATT) * tf.math.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.

def l1lattice(l=0.01):
    return L1L2Lattice(l1=l)


def l2lattice(l=0.01):
    return L1L2Lattice(l2=l)


def l1_l2lattice(l1=0.01, l2=0.01):
    return L1L2Lattice(l1=l1, l2=l2)


def report_model(modeltostudy, history, X, Y, idx=[0]):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.log(history.history['loss']), label='loss')
    plt.plot(np.log(history.history['val_loss']), label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot((history.history['accuracy']), label='acc')
    plt.plot((history.history['val_accuracy']), label='val_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/cluster/results/Accuracy.png')

    W = modeltostudy.layers[-1].weights[0]
    nfilters = W.shape[-1]

    figure_width = 10
    figure_height = 10

    num_rows = 4
    num_cols = 4

    for i in idx:
        plt.figure()
        Z = modeltostudy(X[i:(i + 1)])
        plt.imshow(X[i])
        plt.axis('off')
        plt.title('Original Image')
        plt.savefig('/cluster/results/Original_image.png')

        plt.figure(figsize=(figure_width, figure_height))
        res = []
        for ki in range(nfilters):
            plt.subplot(num_rows, num_cols, ki + 1)
            plt.imshow(Z[0, :, :, ki])
            plt.title(np.mean(Z[0, :, :, ki]), fontsize=8)
            plt.axis('off')
            res.append(np.mean(Z[0, :, :, ki]))
        plt.suptitle('Output filters')
        plt.savefig('/cluster/results/Output_filter.png')

    # Define the number of rows and columns
    num_rows = 4
    num_cols = 4
    plt.figure(figsize=(figure_width, figure_height))
    # Loop through the filters and create subplots in a grid
    for i in range(nfilters):
        plt.subplot(num_rows, num_cols, i + 1)  # Specify row, column, and index
        plt.imshow(W[:, :, 0, i])
        plt.title((np.round(np.max(W[:, :, 0, i])), np.round(np.min(W[:, :, 0, i]))), fontsize=8)
        plt.axis('off')
    plt.suptitle('Filters')
    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    plt.savefig('/cluster/results/Filters.png')

    plt.figure(figsize=(figure_width, figure_height))
    Z = modeltostudy(X)
    Z = np.mean(Z, axis=1)
    Z = np.mean(Z, axis=1)
    l = []
    for i in range(nfilters):
        l.append('Filter' + str(i))
    l.append('Y')
    df = pd.DataFrame(np.concatenate([Z, Y], axis=-1), columns=l)
    df.boxplot(by='Y')

    # Adjust the font size of x-axis and y-axis labels
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.savefig('/cluster/results/Boxplot.png')


from tensorflow.python.keras import activations


class Dilation2D(tf.keras.layers.Layer):
    """
    Sum of Depthwise (Marginal) Dilation 2D on the third axes
    for now assuming channel last

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    Example:
    --------
    from keras.models import Sequential, Model
    from layers import Input
    xin = Input(shape=(28, 28, 3))
    x = Dilation2D(num_filters=16, kernel_size=(5, 5))(xin)
    model = Model(xin, x)
    """

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1, 1), activation=None, use_bias=False,
                 kernel_initializer=tf.keras.initializers.RandomNormal(
                     mean=-.5, stddev=0.1, seed=None),
                 kernel_constraint=None, kernel_regularization=None, bias_initializer='zeros', bias_regularizer=None,
                 bias_constraint=None, **kwargs):
        super(Dilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates = dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel', constraint=self.kernel_constraint,
                                      regularizer=self.kernel_regularization)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.num_filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        # Be sure to call this at the end
        super(Dilation2D, self).build(input_shape)

    def call(self, x):
        res = []
        for i in range(self.num_filters):
            ## erosion2d returns image of same size as x
            ## so taking max over channel_axis
            res.append(
                tf.reduce_sum(dilation2d(x, self.kernel[..., i], self.strides, self.padding), axis=-1))
        output = tf.stack(res, axis=-1)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


@tf.function
def dilation2d(x, st_element, strides, padding, rates=(1, 1)):
    """
    Basic Dilation Operator
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(x, st_element, (1,) + strides + (1,), padding.upper(), "NHWC", (1,) + rates + (1,))
    return x

def Dist_sub(points_image,contour_image):

    points_image = np.clip(points_image, 0, 255).astype(np.uint8)
    points_image = np.squeeze(points_image)
    contours, _ = cv2.findContours(points_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points_coordinates = [tuple(cnt[0][0]) for cnt in contours]
    corresponding_values = [contour_image[y, x] for x, y in points_coordinates]
    reaverage_value = np.mean(corresponding_values) / 255
    return reaverage_value


outputs_train = []
outputs_val = []

for i in range(len(Xtrain)):
    output_i = Dist_sub(Xtrain[i], X2train[i])  # Apply F to the i-th pair of images
    outputs_train.append(output_i)

for i in range(len(Xval)):
    output_i = Dist_sub(Xval[i], X2val[i])  # Apply F to the i-th pair of images
    outputs_val.append(output_i)

outputs_train = np.array(outputs_train)
outputs_val = np.array(outputs_val)


outputs2_train = []
outputs2_val = []

for i in range(len(Xtrain)):
    output2_i = Dist_sub(Xtrain[i], X3train[i])  # Apply F to the i-th pair of images
    outputs2_train.append(output2_i)

for i in range(len(Xval)):
    output2_i = Dist_sub(Xval[i], X3val[i])  # Apply F to the i-th pair of images
    outputs2_val.append(output2_i)

outputs2_train = np.array(outputs2_train)
outputs2_val = np.array(outputs2_val)


outputs3_train = []
outputs3_val = []

for i in range(len(Xtrain)):
    output3_i = Dist_sub(Xtrain[i], X4train[i])  # Apply F to the i-th pair of images
    outputs3_train.append(output3_i)

for i in range(len(Xval)):
    output3_i = Dist_sub(Xval[i], X4val[i])  # Apply F to the i-th pair of images
    outputs3_val.append(output3_i)

outputs3_train = np.array(outputs3_train)
outputs3_val = np.array(outputs3_val)


outputs4_train = []
outputs4_val = []

for i in range(len(Xtrain)):
    output4_i = Dist_sub(Xtrain[i], X5train[i])  # Apply F to the i-th pair of images
    outputs4_train.append(output4_i)

for i in range(len(Xval)):
    output4_i = Dist_sub(Xval[i], X5val[i])  # Apply F to the i-th pair of images
    outputs4_val.append(output4_i)

outputs4_train = np.array(outputs4_train)
outputs4_val = np.array(outputs4_val)

def get_model(model_type='CNN', image_size=256, nfilters=16, num_classes=8, regulatization_term=.001, dist = 1):
    inputs = Input((image_size, image_size, 1))
    inputs2 = Input((dist,))
    inputs3 = Input((dist,))
    inputs4 = Input((dist,))
    inputs5 = Input((dist,))
    ## AUGMENTATION
    # aug =  layers.RandomFlip()(inputs)

    aug = inputs
    #
    if model_type == 'CNN':
        xConv = layers.Conv2D(nfilters, kernel_size=(15, 15), activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=regulatization_term,
                                                                             l2=regulatization_term))(aug)
    elif model_type == 'Dilation':
        xConv = Dilation2D(NFILTERS, kernel_size=(15, 15),
                           kernel_regularization=l1_l2lattice(l1=regulatization_term, l2=regulatization_term))(aug)
    x = layers.GlobalAvgPool2D()(xConv)
    #tf.print("x in model", x)
    x = tf.concat([x, inputs2, inputs3, inputs4, inputs5], axis=- 1)
    #tf.print("concatenate", x)
    x = layers.LayerNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model([inputs, inputs2, inputs3, inputs4, inputs5], outputs)
    modelInterpretation = tf.keras.Model(inputs, xConv)
    return model, modelInterpretation


# historyCNN = modelCNN.fit(Xtrain,Ytrain,validation_data=(Xval,Yval),epochs=NEPOCHS, callbacks=[model_checkpoint_callback])
# modelCNN.load_weights(checkpoint_filepath)


IMAGE_SIZE = 250
NUM_CLASSES = 8
NFILTERS = 16
NEPOCHS = 2000
REGULARIZATION_TERM = 0.01
LEARNING_RATE = 0.0001
dist = 1

modelDIL, modelDILInterpretation = get_model(model_type='Dilation',
                                             image_size=IMAGE_SIZE,
                                             nfilters=NFILTERS,
                                             num_classes=NUM_CLASSES,
                                             regulatization_term=REGULARIZATION_TERM,
                                             dist=1)
modelDIL.summary()

optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
modelDIL.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define the absolute path for checkpoint directory
checkpoint_dir = '/cluster/results/dilation_checkpoint/'

# Ensure the directory exists, create if not
os.makedirs(checkpoint_dir, exist_ok=True)

# Define ModelCheckpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_checkpoint.h5'),  # Use os.path.join for path concatenation
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False
)

historyDIL = modelDIL.fit([Xtrain, outputs_train, outputs2_train, outputs3_train, outputs4_train], Ytrain, batch_size=16, validation_data=([Xval, outputs_val, outputs2_val, outputs3_val, outputs4_val], Yval), epochs=NEPOCHS,
                          callbacks=[model_checkpoint_callback])

predictions = modelDIL.predict([Xval, outputs_val, outputs2_val, outputs3_val, outputs4_val])
predicted_classes = np.argmax(predictions, axis=1)

reshaped_Yval = Yval.reshape(predicted_classes.shape)
confusion_mat = confusion_matrix(reshaped_Yval, predicted_classes)

classes = ["cell_edge", "extranuclear", "foci", "intranuclear", "nuclear_edge", "pericellular", "perinuclear", "random"]

# classes = ["cell_edge", "extranuclear"]  # Replace with your class names
plt.figure(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45, ha="right", fontsize=6)
plt.show()

# Save the figure as an image
plt.savefig('/cluster/results/confusion_matrix.png')

report_model(modelDILInterpretation, historyDIL, Xval, Yval, idx=[0, 1])




