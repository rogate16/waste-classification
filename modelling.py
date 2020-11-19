from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

# Set Path
train_path = "DATASET/TRAIN/"
test_path = "DATASET/TEST/"

# Create Image Generator
train_data = ImageDataGenerator(rescale=1.0/255, width_shift_range=0.1, height_shift_range=0.1,
                                horizontal_flip=True, vertical_flip=True, zoom_range=0.2,
                                shear_range=0.1).\
                flow_from_directory(test_path, target_size=(200,200), class_mode="binary")

test_data = ImageDataGenerator(rescale=1.0/255, width_shift_range=0.1, height_shift_range=0.1,
                                horizontal_flip=True, vertical_flip=True, zoom_range=0.2,
                                shear_range=0.1).\
                flow_from_directory(test_path, target_size=(200,200), class_mode="binary")

# Define Callbacks
reduceLR = ReduceLROnPlateau(monitor="loss", patience=3, mode="min")
early_stop = EarlyStopping(monitor="loss", patience=5, mode="min", restore_best_weights=True)

# Build Model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(200,200,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Fit Model
model.fit_generator(train_data, epochs=100, callbacks=[reduceLR, early_stop])

model.save("model/model.h5", save_format=".h5")