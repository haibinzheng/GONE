"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import numpy as np
from keras import backend as K
from absl import app
from absl import flags
from keras.models import load_model
import tensorflow as tf
import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from keras.optimizers import adam
from keras import regularizers
from sympy import *
from sympy.abc import x,y,a,b,c,d,e,f,g,h
import math
from tensorflow.examples.tutorials.mnist import input_data

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000
img_rows = 32
img_cols = 32
batch_size = 10
FLAGS = flags.FLAGS
epochs = 50
flags.DEFINE_integer(
    "target_epochs", 30, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 30, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 1, "Number of epochs to train attack models.")
epi = 0.001
epi2 = 0.002


def softmax_defend(x):
    return np.exp(epi*x) / np.sum(np.exp(epi*x))


def get_data():
    """Prepare CIFAR10 data."""
    X_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    # X_train /= 255
    # X_test /= 255
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (5, 5),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS)
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation=None))
    model.add(Activation("softmax"))
    Adam = adam(0.001)
    model.compile(Adam, loss="categorical_crossentropy", metrics=["accuracy"])
    # model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def shadow_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (5, 5),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation=None))
    model.add(Activation("softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(NUM_CLASSES,), kernel_regularizer = regularizers.l2(0.000001)))
    #
    model.add(Dropout(0.4, noise_shape=None, seed=None))
    # model.add(Dense(64, activation="relu"))
    # model.add(Dropout(0.2, noise_shape=None, seed=None))
    # model.add(Dense(64, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    # , kernel_regularizer = regularizers.l2(0.000001)
    Adam = adam(0.01)
    model.compile(Adam, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):
    del argv  # Unused.

    (X_train, y_train), (X_test, y_test) = get_data()#训练集，测试集
    targer_x_train = X_train[:10000]
    target_y_train = y_train[:10000]
    # # Train the target model.
    print("Training the target model...")
    # target_model = target_model_fn()
    # target_model.fit(
    #     targer_x_train, target_y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    # )
    #
    # target_model.save('cifar10_model.h5')
    target_model = load_model('cifar10_model.h5')

    # model = tf.keras.models.load_model('model.h5', compile=False)
    # Train the shadow models.
    smb = ShadowModelBundle(
        shadow_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    # X_shadow, y_shadow = smb.fit_transform(
    #     attacker_X_train,
    #     attacker_y_train,
    #     fit_kwargs=dict(
    #         epochs=FLAGS.target_epochs,
    #         verbose=True,
    #         validation_data=(attacker_X_test, attacker_y_test),
    #     ),
    # )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    # x_shadow = X_shadow[:, : NUM_CLASSES]
    # y_shadow = keras.utils.to_categorical(y_shadow, 2)
    amb = attack_model_fn()
    # Fit the attack models.
    print("Training the attack models...")
    # amb.fit(x_shadow, y_shadow,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1)
    # amb.save('cifar10_amb_model.h5')
    amb = load_model('cifar10_amb_model.h5')

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_test_data = attack_test_data[:, :NUM_CLASSES]
    attack_test_data2 = attack_test_data
    for i in range(attack_test_data.shape[0]):
        attack_test_data[i] = softmax_defend(attack_test_data[i])

    for i in range(attack_test_data.shape[0]):
        attack_test_data2[i] = adaptive_attack(attack_test_data[i])

    attack_guesses = amb.predict(attack_test_data)
    attack_guesses2 = []
    attack_guesses2 = np.argmax(attack_guesses, axis=1)
    # for i in attack_guesses:
    #     if i[0] > 0.5:
    #         attack_guess = 0
    #     else:
    #         attack_guess = 1
    #     attack_guesses2.append(attack_guess)
    # attack_guesses2 = np.array(attack_guesses2)
    attack_accuracy = np.mean(attack_guesses2 == real_membership_labels)

    tp = 0#正正
    fn = 0#正负
    fp = 0#负正
    tn = 0#负负
    for i in range(ATTACK_TEST_DATASET_SIZE):
        if attack_guesses2[i] == real_membership_labels[i]:
            tp = tp + 1
        else:
            fn = fn + 1

    for i in range(ATTACK_TEST_DATASET_SIZE, 2 * ATTACK_TEST_DATASET_SIZE):
        if attack_guesses2[i] == real_membership_labels[i]:
            tn = tn + 1
        else:
            fp = fp + 1
    # Compute the attack precision.

    precision = tp / (tp + fp + 1)

    # Compute the attack recall.

    recall = tp / (tp + fn + 1)



    print(attack_accuracy)
    print(recall)
    print(precision)



if __name__ == "__main__":
    app.run(demo)
