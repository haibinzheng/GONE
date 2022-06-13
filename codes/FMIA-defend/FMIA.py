"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import numpy as np

from absl import app
from absl import flags
from keras.models import load_model
import tensorflow as tf
import keras
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from mia.estimators2 import ShadowModelBundle, AttackModelBundle, prepare_attack_data_target, prepare_attack_data2, prepare_attack_data
from keras.optimizers import adam
from keras import regularizers
from keras import backend as K
NUM_CLASSES = 10
WIDTH = 28
HEIGHT = 28
CHANNELS = 1
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 30, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 50, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 1, "Number of epochs to train attack models.")
img_rows = 28
img_cols = 28
def get_data():
    """Prepare CIFAR10 data."""
    # X_train = np.load('/home/NewDisk/sgwc/dataset_mnist/x_train.npy')
    # y_train = np.load('/home/NewDisk/sgwc/dataset_mnist/y_train.npy')
    # X_test = np.load('/home/NewDisk/sgwc/dataset_mnist/x_test.npy')
    # y_test = np.load('/home/NewDisk/sgwc/dataset_mnist/y_test.npy')
    mnist = input_data.read_data_sets("/home/NewDisk/mnist/")
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
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
            input_shape=(WIDTH, HEIGHT, CHANNELS),kernel_regularizer=regularizers.l2(0.05)
        )
    )
    # model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.05)))
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation="tanh"))
    # model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation=None))
    model.add(Activation("softmax"))

    #no defense
    # model.add(layers.Dense(NUM_CLASSES))#be defended
    # model.add(Lambda(softmax_defend()))


    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(NUM_CLASSES,)))
    # , kernel_regularizer = regularizers.l2(0.000001)
    # model.add(Dropout(0.4, noise_shape=None, seed=None))
    # model.add(Dense(64, activation="relu"))
    # model.add(Dropout(0.2, noise_shape=None, seed=None))
    # model.add(Dense(64, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    Adam=adam(0.001)
    model.compile(Adam, loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):
    del argv  # Unused.
    # Norm = np.random.normal(loc=0, scale=0.005, size=10),  # loc 表示均值 scale 表示标准差σ size 表示生成个数

    (X_train, y_train), (X_test, y_test) = get_data()#训练集，测试集
    targer_x_train = X_train[:4600]
    target_y_train = y_train[:4600]
    # Train the target model.
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        targer_x_train, target_y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    )
    # target_model.evaluate(X_test, y_test)
    # target_model.save('cifar10_model/15000_cnn.h5')
    # target_model = load_model('cifar10_model/15000_cnn.h5')

    # model = tf.keras.models.load_model('model.h5', compile=False)
    # Train the shadow models.
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)
    tsne_x = X_shadow[:, :NUM_CLASSES]
    tsne_y = X_shadow[:, NUM_CLASSES:]
    np.save('tsne_x_MNIST', tsne_x)
    np.save('tsne_y_MNIST', tsne_y)
    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels, c = prepare_attack_data2(
        target_model, data_in, data_out
    )


    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    tp = 0  # 正正
    fn = 0  # 正负
    fp = 0  # 负正
    tn = 0  # 负负
    for i in range(ATTACK_TEST_DATASET_SIZE):
        if attack_guesses[i] == real_membership_labels[i]:
            tp = tp + 1
        else:
            fn = fn + 1

    for i in range(ATTACK_TEST_DATASET_SIZE, 2 * ATTACK_TEST_DATASET_SIZE):
        if attack_guesses[i] == real_membership_labels[i]:
            tn = tn + 1
        else:
            fp = fp + 1
    # Compute the attack precision.

    precision = tp / (tp + fp)

    # Compute the attack recall.

    recall = tp / (tp + fn)

    print(attack_accuracy)
    print(precision)
    print(recall)
    print(c)


if __name__ == "__main__":
    app.run(demo)
