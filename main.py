import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import dill


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Embedding(input_dim=input_shape, output_dim=64),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Sample data
# texts = ["I love this movie", "This movie is terrible",
#         "Great experience", "Poor service",]
# 1 for positive sentiment, 0 for negative sentiment
# labels = np.array([1, 0, 1, 0])
texts = []
labels = []

data = []

positive_files = []
for root, _, files in os.walk("./aclImdb/train/pos"):
    for file in files:
        pt = os.path.join(root, file)
        if pt in positive_files:
            continue
        positive_files.append(pt)

for file in positive_files:
    with open(file, "r") as f:
        data.append((1, f.read().replace("<br />", "\n")))

negative_files = []
for root, _, files in os.walk("./aclImdb/train/neg"):
    for file in files:
        pt = os.path.join(root, file)
        if pt in negative_files:
            continue
        negative_files.append(pt)

for file in negative_files:
    with open(file, "r") as f:
        data.append((0, f.read().replace("<br />", "\n")))


random.shuffle(data)
for d in data:
    labels.append(d[0])
    texts.append(d[1])

labels = np.array(labels)

# Tokenize the texts
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
with open("tokenizer", "wb") as f:
    dill.dump(tokenizer, f)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
num_words = len(word_index) + 1

# Padding sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)


# Create and train the model
model = create_model(input_shape=num_words, num_classes=2)
while True:
    model.fit(X_train, y_train, epochs=1, batch_size=2, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
