# CELEBRITIES_IMAGE_CLASSIFICATION
## DOCUMENTATION
I have chosen a Convolutional Neural Network (CNN) for multi-class classification. The model architecture is as follows:
model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
•	Input Layer: Data augmentation using RandomFlip, RandomRotation, and RandomZoom.
•	Convolutional Layer: 32 filters with a kernel size of (3, 3), ReLU activation.
•	MaxPooling Layer: Reduces spatial dimensions.
•	Flatten Layer: Flattens the output for dense layers.
•	Dense Layers: Two fully connected layers with ReLU activation.
•	Output Layer: Dense layer with num_classes units and softmax activation for multi-class classification.

## Training Process:
•	The model is compiled with Adam optimizer and Categorical Crossentropy loss. The training process is executed with the following code:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_train_one_hot = to_categorical(y_train, num_classes=5)
history = model.fit(x_train, y_train_one_hot, epochs=100, batch_size=128, validation_split=0.1)
Critical Findings:
### 1.	Data Preparation:
o	Images are loaded and resized to (128, 128) pixels.
o	Data is split into training and testing sets.
### 2.	Data Augmentation:
o	Utilized RandomFlip, RandomRotation, and RandomZoom for augmenting training data.
### 3.	Normalization:
o	Image data is normalized using tf.keras.utils.normalize.
### 4.	Model Architecture:
o	Convolutional layers followed by dense layers for feature extraction.
o	Dropout is used for regularization.
o	Softmax activation in the output layer for multi-class classification.
### 5.	Training:
o	Model is trained for 100 epochs with a batch size of 128.
### 6.	Evaluation:
o	Model performance is evaluated on the test set using model.evaluate.
o	Classification report is generated to assess precision, recall, and F1-score.



