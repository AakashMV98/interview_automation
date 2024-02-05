import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import json
import pickle
import numpy as np
import random
import tensorflow as tf
from keras.callbacks import EarlyStopping

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Replace 'data.json' with the full path if necessary
data_file_path = 'data.json'

with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

questions = data["questions"]

# Load preprocessed data
with open('texts.pkl', 'rb') as file:
    words = pickle.load(file)
with open('labels.pkl', 'rb') as file:
    classes = pickle.load(file)

documents = []
ignore_words = ['?', '!']

for entry in questions:
    w = nltk.word_tokenize(entry['question'])
    words.extend(w)
    documents.append((w, entry['question']))
    if entry['question'] not in classes:
        classes.append(entry['question'])

words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

with open('texts.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('labels.pkl', 'wb') as file:
    pickle.dump(classes, file)

training = []
output_empty = [0] * len(classes)

max_words = len(words)

for doc in documents:
    bag = [1 if lemmatizer.lemmatize(word.lower()) in doc[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    bag += [0] * (max_words - len(bag))
    output_row += [0] * (len(classes) - len(output_row))

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

def create_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Use validation split during training
    validation_split = 0.1

    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, validation_split=validation_split, callbacks=[early_stopping])

    # Save both model architecture and weights
    model.save('model.h5')
    model.save_weights('model_weights.h5')

    print("Model created")  

create_model(train_x, train_y)
