import visualkeras
from PIL import ImageFont
from tensorflow.keras import datasets, layers, models, optimizers

modelNB1 = models.Sequential()
modelNB1.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28,1)))
modelNB1.add(layers.MaxPooling2D((2, 2)))
modelNB1.add(layers.Flatten())
modelNB1.add(layers.Dense(101, activation='softmax'))


modelNB3 = models.Sequential()
modelNB3.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28,1)))
modelNB3.add(layers.MaxPooling2D((2, 2)))
modelNB3.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelNB3.add(layers.MaxPooling2D((2, 2)))
modelNB3.add(layers.Conv2D(32, (3, 3), activation='relu'))
modelNB3.add(layers.MaxPooling2D((2, 2)))
modelNB3.add(layers.Flatten())
modelNB3.add(layers.Dense(101, activation='softmax'))

model = modelNB1
#font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True,to_file='NB1.png')  # font is optional!

model = modelNB3
#font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True,to_file='NB3.png')  # font is optional!