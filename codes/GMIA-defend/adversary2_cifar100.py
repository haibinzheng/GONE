"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import numpy as np

from absl import app
from absl import flags
from keras.models import load_model
import tensorflow as tf
import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from mia.estimators_defend import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from keras.optimizers import adam
from keras import regularizers
topX = 3
NUM_CLASSES = 100
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000
batch_size = 10
epochs = 50
num_class = 20
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 60, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 1, "Number of epochs to train attack models.")


def clipDataTopX(dataToClip, top=3):
	res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	return np.array(res)


def get_data():
    """Prepare CIFAR100 data."""
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
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)

def get_data_news():
    """Prepare lfw data."""
    targetTrain = np.load('targetTrain.npz')
    targetTest = np.load('targetTest.npz')
    shadowTrain = np.load('shadowTrain.npz')
    shadowTest = np.load('shadowTest.npz')
    targetTrain_x = targetTrain['arr_0']
    targetTrain_y = targetTrain['arr_1']
    targetTest_x = targetTest['arr_0']
    targetTest_y = targetTest['arr_1']
    shadowTrain_x = shadowTrain['arr_0']
    shadowTrain_y = shadowTrain['arr_1']
    shadowTest_x = shadowTest['arr_0']
    shadowTest_y = shadowTest['arr_1']
    targetTrain_y = keras.utils.to_categorical(targetTrain_y)
    targetTest_y = keras.utils.to_categorical(targetTest_y)
    shadowTrain_y = keras.utils.to_categorical(shadowTrain_y)
    shadowTest_y = keras.utils.to_categorical(shadowTest_y)

    # y_test = keras.utils.to_categorical(y_test)
    # X_train = X_train.astype("float32")
    # X_test = X_test.astype("float32")
    # y_train = y_train.astype("float32")
    # y_test = y_test.astype("float32")
    # X_train /= 255
    # X_test /= 255
    return (targetTrain_x, targetTrain_y), (targetTest_x, targetTest_y), (shadowTrain_x, shadowTrain_y), (shadowTest_x, shadowTest_y)


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

    #no defense
    # model.add(layers.Dense(NUM_CLASSES))#be defended
    # model.add(Lambda(softmax_defend()))


    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def shadow_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = Sequential()

    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_class, activation=None))
    model.add(Activation("softmax"))
    # Adam=adam(0.1)
    # model.compile(Adam, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(topX,), kernel_regularizer = regularizers.l2(0.000001)))
    #
    # model.add(Dropout(0.4, noise_shape=None, seed=None))
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

    (X_train, y_train), (X1_test, y1_test) = get_data()#训练集，测试集
    (targetTrain_x, targetTrain_y), (targetTest_x, targetTest_y), (shadowTrain_x, shadowTrain_y), (
        shadowTest_x, shadowTest_y) = get_data_news()

    X_test = np.vstack((targetTrain_x, targetTest_x))
    y_test = np.vstack((targetTrain_y, targetTest_y))
    targer_x_train = X_train[:4600]
    target_y_train = y_train[:4600]
    # Train the target model.
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        targer_x_train, target_y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    )

    # target_model.save('cifar100_model/29540_cnn.h5')
    # target_model = load_model('cifar100_model/4600_cnn.h5')
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
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )
    x_shadow = X_shadow[:, : num_class]
    x_shadow = clipDataTopX(x_shadow, top=topX)
    y_shadow = keras.utils.to_categorical(y_shadow, 2)
    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = attack_model_fn()
    # Fit the attack models.
    print("Training the attack models...")
    # amb.fit(
    #     X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    # )
    amb.fit(x_shadow, y_shadow,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = targer_x_train[:ATTACK_TEST_DATASET_SIZE], target_y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X1_test[:ATTACK_TEST_DATASET_SIZE], y1_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    # attack_guesses = amb.predict(attack_test_data)
    # attack_accuracy = np.mean(attack_guesses == real_membership_labels)
    attack_test_data = attack_test_data[:, :NUM_CLASSES]
    attack_test_data = clipDataTopX(attack_test_data, top=topX)
    attack_guesses = amb.predict(attack_test_data)
    attack_guesses2 = []
    for i in attack_guesses:
        if i[0] > 0.5:
            attack_guess = 0
        else:
            attack_guess = 1
        attack_guesses2.append(attack_guess)
    attack_guesses2 = np.array(attack_guesses2)
    attack_accuracy = np.mean(attack_guesses2 == real_membership_labels)

    tp = 0  # 正正
    fn = 0  # 正负
    fp = 0  # 负正
    tn = 0  # 负负
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

    precision = tp / (tp + fp)

    # Compute the attack recall.

    recall = tp / (tp + fn)

    print(attack_accuracy)
    print(precision)
    print(recall)


if __name__ == "__main__":
    app.run(demo)
