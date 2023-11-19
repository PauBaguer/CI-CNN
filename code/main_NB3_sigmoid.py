import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
import arrow

K = 1

# Input dataset.
mat = scipy.io.loadmat('caltech101_silhouettes_28.mat')
inputs = mat.get('X')
s = inputs.shape
images = np.array([np.reshape(inputs[i,:],[28,28]) for i in range(s[0])])
images = tf.reshape(images,shape=[8671,28,28,1])
images = np.array(images)
labels = np.array(mat.get('Y')).T -1

# Split dataset into train and test.
train_pct = 0.1#0.8#0.4#0.1
valid_pct = 0.1#0.1#0.2#0.1
test_pct =  0.8#0.1#0.4#0.8
train_images,test_images,train_labels,test_labels = train_test_split(images, labels, train_size=train_pct+valid_pct, test_size=test_pct)
#080101
#402040
#010108
figname='010108-softmax-nb3'

def run_K_times(K,train_images,test_images,train_labels,test_labels,learning_rate):
# Run K times.

    train_acc_arr = []
    test_acc_arr = []
    for i in range(0,K):
     
        modelNB3 = models.Sequential()
        modelNB3.add(layers.Conv2D(128, (3, 3), activation='sigmoid', input_shape=(28, 28,1)))
        modelNB3.add(layers.MaxPooling2D((2, 2)))
        modelNB3.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
        modelNB3.add(layers.MaxPooling2D((2, 2)))
        modelNB3.add(layers.Conv2D(32, (3, 3), activation='sigmoid',))
        modelNB3.add(layers.MaxPooling2D((2, 2)))
        modelNB3.add(layers.Flatten())
        modelNB3.add(layers.Dense(101, activation='softmax'))

        model = modelNB3
        opt = optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=opt,
                run_eagerly=True,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        history=None
        history = model.fit(train_images, train_labels, epochs=100, validation_split=valid_pct/(train_pct+valid_pct),batch_size=32)

        # Plot training progress.
        print(history.history['accuracy'])
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='accuracy')
        ax.plot(history.history['val_accuracy'], label = 'val_accuracy')
        ax.set_title(f'Training progress, lambda:{learning_rate}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.grid()

        now =arrow.now().format('HH-mm-ss')
        plt.legend(loc='lower right')
        plt.savefig(f'tensorflow/figs/sigmoid/train/test-{figname}-{now}.png')



    # print(images.shape)
    # print(labels.shape)
    # print(test_images.shape)
    # print(test_labels.shape)

        print("Evaluate train")
        #train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
        train_acc = history.history['accuracy'][-1]
        print("Evaluate test")
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    train_acc_mean = np.mean(train_acc_arr)
    test_acc_mean = np.mean(test_acc_arr)

    print()
    print("===RESULTS===")
    print(f"Train Accuracy: {train_acc_mean*100:2f}%")
    print(train_acc_arr)
    print(f"Test Accuracy: {test_acc_mean*100:2f}%")
    print(test_acc_arr)

    return train_acc_mean,test_acc_mean


def optimize_learning_rate():
    learning_rate_range = np.logspace(-4,-3,5)
    #learning_rate_range = np.linspace(3e-6,1e-5,15)
    train_acc_arr = []
    test_acc_arr = []
    for i, learning_rate in enumerate(learning_rate_range):
        train_acc_mean,test_acc_mean = run_K_times(K,train_images,test_images,train_labels,test_labels,learning_rate)
        train_acc_arr.append(train_acc_mean)
        test_acc_arr.append(test_acc_mean)

    fig, ax = plt.subplots()
    ax.set_xscale('log')

    ax.plot(learning_rate_range,train_acc_arr, label='Trining accuracy')
    ax.plot(learning_rate_range,test_acc_arr, label='Test accuracy')
    ax.set_title(f'Learning rate optimization')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Accuracy')
    ax.grid()

    now =arrow.now().format('HH-mm-ss')
    plt.legend(loc='lower right')
    plt.savefig(f'tensorflow/figs/sigmoid/optimization/learning-rate-{figname}-{now}.png')


def run_optimized():
    K=3
    train_acc_mean,test_acc_mean = run_K_times(K,train_images,test_images,train_labels,test_labels,0.001)


if __name__ == "__main__":
    #optimize_learning_rate()
    run_optimized()