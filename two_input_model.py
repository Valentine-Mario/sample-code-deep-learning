from keras.models import Model
from keras import layers, Input
import numpy as np



text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')

#Embeds the inputs into a sequence of vectors of size 64
embedded_text = layers.Embedding( 64, text_vocabulary_size)(text_input)

#encode the vector into a single vector via LSTM
encoded_text = layers.LSTM(32)(embedded_text)

#same processing with different instances for the questions
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

#concat embedded text and embedded question
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

#a model instanstation specify input and output
model = Model(inputs=[text_input, question_input], outputs=answer)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

num_samples = 1000
max_length = 100

#Generates dummy Numpy data
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

#fit using a list of input
model.fit([text, question], answers, epochs=10, batch_size=128)

#fit using a dict of input
model.fit({'text': text, 'question': question}, answers,epochs=10, batch_size=128)