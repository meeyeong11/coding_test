# -*- coding: utf-8 -*-

import numpy as np
import time
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(xf, yf):
    X = [['start']+line.split()+['end'] for line in open(xf).readlines()]#[:3]
    Y = [['start']+line.split()+['end'] for line in open(yf).readlines()]#[:3]

    return X, Y

data_dir = './data/'

trxf = data_dir + 'train_source.txt'
tryf = data_dir+'train_target.txt'

X,Y = load_data(trxf, tryf)
X = X[:len(Y)]
max_in_len = max([len(line) for line in X])
max_out_len = max([len(line) for line in Y])

val_X = X[:int(len(X)/10)]
val_Y = Y[:int(len(X)/10)]

def get_vocab(X):
    from collections import defaultdict
    vocab_idx = defaultdict(int)

    for line in X:
        #print(line)
        for tok in line:
            if tok not in vocab_idx:
                vocab_idx[tok] = len(vocab_idx)
                #print(tok, len(vocab_idx))
    return vocab_idx

# get data pad seq
x_vocab_idx =  get_vocab(X)
y_vocab_idx =  get_vocab(Y)

x_tokenizer = Tokenizer(num_words=len(x_vocab_idx))
x_tokenizer.fit_on_texts(X)
x_tr = x_tokenizer.texts_to_sequences(X)
x_val = x_tokenizer.texts_to_sequences(val_X)
x_tr = pad_sequences(x_tr, maxlen=max_in_len, padding='post')
x_val = pad_sequences(x_val, maxlen=max_in_len, padding='post')

y_tokenizer = Tokenizer(num_words=len(y_vocab_idx))
y_tokenizer.fit_on_texts(Y)
y_tr = y_tokenizer.texts_to_sequences(Y)
y_val = y_tokenizer.texts_to_sequences(val_Y)
y_tr = pad_sequences(y_tr, maxlen=max_out_len, padding='post')
y_val = pad_sequences(y_val, maxlen=max_out_len, padding='post')


print(x_tr.shape, y_tr.shape)
print(x_val.shape, y_val.shape)

#Encoder
h_dim = 20
dropout_r = 0.4
enc_input = Input(shape=(max_in_len), name='enc-in')
enc_emb = Embedding(len(x_vocab_idx)+1,len(x_vocab_idx) , trainable=True,  name='enc-emb')(enc_input)
enc_lstm = LSTM(h_dim, return_sequences=True, return_state=True, dropout=dropout_r, recurrent_dropout=dropout_r, name='enc-lstm')
enc_outputs, enc_h, enc_c = enc_lstm(enc_emb)

#Decoder
#dec_emb_dim = 30
dec_input = Input(shape=(None,), name='dec-in')
dec_emb_layer = Embedding(len(y_vocab_idx)+1, len(y_vocab_idx), trainable=True, name='dec-emb')
dec_emb = dec_emb_layer(dec_input)
dec_lstm = LSTM(h_dim, return_sequences=True, return_state=True, dropout=dropout_r, recurrent_dropout=dropout_r, name='dec-lstm')
dec_lstm_output, dec_h, dec_c = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])

dec_dense = TimeDistributed(Dense(len(y_vocab_idx)+1, activation='softmax', name='dec-dense'))
dec_output = dec_dense(dec_lstm_output, 'dec-output')

model = Model([enc_input, dec_input], dec_output)
model.summary()
adam = optimizers.Adam(lr=0.01)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=1)
history = model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=1,\
                    callbacks=[es],batch_size=64, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

save_dir = data_dir.replace('data','save')
checkpoint_path = save_dir+'simple_lstm-h_dim-'+str(h_dim)+'-drop_r='+str(dropout_r)+'.ckpt'
model.save_weights(checkpoint_path.format(epoch=0))

#save_model(model, save_dir+'emb_dim-'+str(emb_dim)+'h_dim-'+str(h_dim)+'drop_r='+str(dropout_r)+'_model.md')
# get test data
testxf = data_dir + 'test_source.txt'
testyf = data_dir+'test_target.txt'

X,Y = load_data(testxf, testyf)

x_test = x_tokenizer.texts_to_sequences(X)
x_test = pad_sequences(x_test, maxlen=max_in_len, padding='post')

y_test = y_tokenizer.texts_to_sequences(Y)
y_test = pad_sequences(y_test, maxlen=max_out_len, padding='post')
print(x_test.shape, y_test.shape)

# get model to predict
enc_model = Model(inputs=enc_input, outputs=[enc_outputs,enc_h, enc_c])
dec_in_h = Input(shape=(h_dim,))

dec_in_c = Input(shape=(h_dim,))
dec_h_in = Input(shape=(max_in_len, h_dim))
dec_emb2 = dec_emb_layer(dec_input)
dec_lstm_output2, dec_h2, dec_c2 = dec_lstm(dec_emb2, initial_state=[dec_in_h, dec_in_c])
dec_output2 = dec_dense(dec_lstm_output2)
dec_model = Model([dec_input]+[dec_h_in, dec_in_h, dec_in_c], [dec_output2]+[dec_h2, dec_c2])


reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

def decode_seq(in_seq):

  enc_out, enc_h, enc_c = enc_model.predict(in_seq)
  
  target_seq = np.zeros((1,1))
  target_seq[0,0] = target_word_index['start']

  stop_cond = False
  decoded_sent = ''
  decoded_idxs = []

  while not stop_cond:
    output_tok, h, c = dec_model.predict([target_seq]+[enc_out, enc_h, enc_c])
    output_tok_idx = np.argmax(output_tok[0,-1,:])
    output_tok = reverse_target_word_index[output_tok_idx]

    if (output_tok != 'end'):
      decoded_sent += output_tok 
      decoded_idxs.append(output_tok_idx)

    if (output_tok == 'end' or len(decoded_idxs)>max_out_len):
      stop_cond = True 

    target_seq = np.zeros((1,1))
    target_seq[0,0] = output_tok_idx 
    enc_h, enc_c = h, c

  return decoded_idxs


st = time.time()
y_pred = []
for i, x in enumerate(x_test):#[:10]:
  y = decode_seq(x.reshape(1,max_in_len))
  y_pred.append(y)
  if i%100 ==0:
    print(i, (time.time()-st)/60)

res_dir = data_dir.replace('data', 'results')

res_f = res_dir+'simple_lstm-h_dim-'+str(h_dim)+'-drop_r='+str(dropout_r)+'.res.txt'
with open(res_f, 'w') as wf:
  wf.write('\n'.join([' '.join([str(idx) for idx in l])+' | '+str(list(y)).replace('0, ', '').replace(',','').replace('[', '').replace('0]', '') for l,y in zip(y_pred, y_test)]))

