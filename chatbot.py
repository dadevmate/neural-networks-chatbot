import tensorflow as tf
import numpy as np


input_sequences = ["hi", "how are you", "what is your name", "bye"]
output_sequences = ["hello", "I'm fine", "I'm a chatbot", "goodbye"]

input_vocab = set(" ".join(input_sequences).split())
output_vocab = set(" ".join(output_sequences).split())
vocab = input_vocab.union(output_vocab)
vocab_size = len(vocab)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

input_data = [[word2idx[word] for word in sentence.split()] for sentence in input_sequences]
output_data = [[word2idx[word] for word in sentence.split()] for sentence in output_sequences]

encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, 64)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, 64)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit([np.array(input_data), np.array(output_data[:, :-1])],
          np.expand_dims(output_data[:, 1:], -1),
          epochs=50, batch_size=16)

encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(64,))
decoder_state_input_c = tf.keras.layers.Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

def generate_response(input_text):
    input_seq = [word2idx[word] for word in input_text.lower().split()]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=len(input_seq), padding='post')

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx['start']

    stop_condition = False
    response = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2word[sampled_token_index]
        response.append(sampled_word)

        if sampled_word == 'end' or len(response) > 20:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return ' '.join(response)


while True:
    user_input = input("User: ")
    if user_input.lower() == 'bye':
        break

    response = generate_response(user_input)
    print("Chatbot:", response)
