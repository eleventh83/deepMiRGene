from __future__ import print_function
import rlcompleter, readline
readline.parse_and_bind('tab:complete')
import numpy as np
np.random.seed(1337)
from os.path import exists
from os import makedirs
from Bio import SeqIO  ## fasta read
import RNA ## RNAFold
import re ## reg to find loops
from sklearn.model_selection import KFold  ## for cv
from sklearn import metrics ## evaluation


# keras
from keras.models import Model
from keras.layers import Input, LSTM, TimeDistributed, Dropout, Dense, Permute, Flatten, Multiply, RepeatVector, Activation, Masking
from keras import regularizers, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.wrappers import Wrapper
from keras.engine.topology import InputSpec
from keras import backend as K


def import_seq(filename):
 seqs = []
 for record in SeqIO.parse(filename, "fasta"):
  a_seq = str(record.seq)
  seqs.append(a_seq)
 return seqs


def seq2str(seqs):
 return [RNA.fold(a_seq)[0] for a_seq in seqs]


def seq2num(a_seq):
 ints_ = [0]*len(a_seq)
 for i, c in enumerate(a_seq.lower()):
  if c == 'c':
   ints_[i] = 1
  elif c == 'g':
   ints_[i] = 2
  elif (c == 'u') | (c == 't'):
   ints_[i] = 3
  
 return ints_


def str2num(a_str):
 ints_ = [0]*len(a_str)
 for i, c in enumerate(a_str.lower()):
  if c == ')':
   ints_[i] = 1
  elif c == '.':
   ints_[i] = 2
  elif c == ':':
   ints_[i] = 3
  
 return ints_



# convert loops from '.'s to ':'s
def convloops(a_str):
 chrs_ = a_str[:]
 prog = re.compile('\(+(\.+)\)+')
 for m in prog.finditer(a_str):
  #print m.start(), m.group()
  chrs_ = "".join((chrs_[:m.regs[1][0]],':'*(m.regs[1][1]-m.regs[1][0]),chrs_[(m.regs[1][1]):]))
 
 return chrs_



def encode(seqs,strs):
 if not isinstance(seqs, list):
  print("[ERROR:encode] Input type must be multidimensional list.")
  return
 
 if len(seqs) != len(strs) :
  print("[ERROR:encode] # sequences must be equal to # structures.")
  return
 
 encs = []
 for a_seq, a_str in zip(seqs,strs):
  encs.append([4*i_seq+i_str+1 for i_seq, i_str in zip(seq2num(a_seq),str2num(convloops(a_str)))])
 
 return encs



def import_data(filename):
 encs = []
 seqs = import_seq(filename)
 strs = seq2str(seqs)
 return encode(seqs,strs)



def one_hot_wrap(X_encs, MAX_LEN, DIM_ENC):
 num_X_encs = len(X_encs)
 X_encs_padded = pad_sequences(X_encs, maxlen = MAX_LEN, dtype='int8')
 X_encs_ = np.zeros(num_X_encs).tolist()
 for i in range(num_X_encs):
  X_encs_[i] = one_hot(X_encs_padded[i],DIM_ENC)
 
 return np.int32(X_encs_)



def one_hot(X_enc,DIM_ENC):
 X_enc_len = len(X_enc)
 X_enc_vec = np.zeros((X_enc_len, DIM_ENC))
 X_enc_vec[np.arange(np.nonzero(X_enc)[0][0],X_enc_len), np.int32([X_enc[k]-1 for k in np.nonzero(X_enc)[0].tolist()])] = 1
 
 return X_enc_vec.tolist()



def splitCV(data, kfold):
 kf = KFold(n_splits=kfold, shuffle=True)
 train_idx = []
 test_idx = []
 for train, test in kf.split(data):
  train_idx.append(train)
  test_idx.append(test)
 
 return train_idx, test_idx




def perfeval(predictions, Y_test, verbose=0):
 class_label = np.uint8(np.argmax(predictions,axis=1))
 R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))
 CM = metrics.confusion_matrix(R, class_label, labels=None)
 
 CM = np.double(CM)
 acc = (CM[0][0]+CM[1][1])/(CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1])
 se = (CM[0][0])/(CM[0][0]+CM[0][1])
 sp = (CM[1][1])/(CM[1][0]+CM[1][1])
 f1 = (2*CM[0][0])/(2*CM[0][0]+CM[0][1]+CM[1][0])
 ppv = (CM[0][0])/(CM[0][0]+CM[1][0])
 mcc = (CM[0][0]*CM[1][1]-CM[0][1]*CM[1][0])/np.sqrt((CM[0][0]+CM[0][1])*(CM[0][0]+CM[1][0])*(CM[0][1]+CM[1][1])*(CM[1][0]+CM[1][1]))
 gmean = np.sqrt(se*sp)
 auroc = metrics.roc_auc_score(Y_test[:,0],predictions[:,0])
 aupr = metrics.average_precision_score(Y_test[:,0],predictions[:,0],average="micro")
 
 if verbose == 1:
  print("SE:","{:.3f}".format(se),"SP:","{:.3f}".format(sp),"F-Score:","{:.3f}".format(f1), "PPV:","{:.3f}".format(ppv),"gmean:","{:.3f}".format(gmean),"AUROC:","{:.3f}".format(auroc), "AUPR:","{:.3f}".format(aupr))
 
 return [se,sp,f1,ppv,gmean,auroc,aupr,CM]


def wrtrst(filehandle, rst, nfold=0, nepoch=0):
 filehandle.write(str(nfold+1)+" "+str(nepoch+1)+" ")
 filehandle.write("SE: %s SP: %s F-score: %s PPV: %s g-mean: %s AUROC: %s AUPR: %s\n" %
 ("{:.3f}".format(rst[0]),
 "{:.3f}".format(rst[1]),
 "{:.3f}".format(rst[2]),
 "{:.3f}".format(rst[3]),
 "{:.3f}".format(rst[4]),
 "{:.3f}".format(rst[5]),
 "{:.3f}".format(rst[6])))
 filehandle.flush()
 return


def make_safe(x):
 return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)



class ProbabilityTensor(Wrapper):
 def __init__(self, dense_function=None, *args, **kwargs):
  self.supports_masking = True
  self.input_spec = [InputSpec(ndim=3)]
  layer = TimeDistributed(Dense(1, name='ptensor_func'))
  super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)
 
 def build(self, input_shape):
  assert len(input_shape) == 3
  self.input_spec = [InputSpec(shape=input_shape)]
  if K._BACKEND == 'tensorflow':
   if not input_shape[1]:
    raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')
   
  if not self.layer.built:
   self.layer.build(input_shape)
   self.layer.built = True
  
  super(ProbabilityTensor, self).build()
 
 def compute_output_shape(self, input_shape):
  if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
   input_shape = input_shape[0]
  
  return (input_shape[0], input_shape[1])
 
 def squash_mask(self, mask):
  if K.ndim(mask) == 2:
   return mask
  elif K.ndim(mask) == 3:
   return K.any(mask, axis=-1)
 
 def compute_mask(self, x, mask=None):
  if mask is None:
   return None
  return self.squash_mask(mask)
 
 def call(self, x, mask=None):
  energy = K.squeeze(self.layer(x), 2)
  p_matrix = K.softmax(energy) ## (nb_sample, time)
  if mask is not None:
   mask = self.squash_mask(mask)
   p_matrix = make_safe(p_matrix * mask)
   p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
  return p_matrix
 
 def get_config(self):
  config = {}
  base_config = super(ProbabilityTensor, self).get_config()
  return dict(list(base_config.items()) + list(config.items()))




class SoftAttention(ProbabilityTensor):
 def compute_output_shape(self, input_shape):
  return [(input_shape[0], input_shape[1] * input_shape[2]), (input_shape[0], input_shape[1])]
 
 def compute_mask(self, x, mask=None):
  if mask is None or mask.ndim==2:
   return [None, None]
  else:
   raise Exception("Unexpected situation")
 
 def call(self, x, mask=None):
  p_vector = super(SoftAttention, self).call(x, mask)
  p_vectors = K.expand_dims(p_vector, 2)
  expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
  mul = expanded_p * x
  
  return [K.reshape(mul,[K.shape(x)[0],K.shape(x)[1]*K.shape(x)[2]]), p_vector]




### model definition
def mymodel(MAX_LEN,DIM_ENC,DIM_LSTM1,DIM_LSTM2,DIM_DENSE1,DMI_DENSE2):
 inputs = Input(shape=(MAX_LEN,DIM_ENC), name='inputs')
 msk = Masking(mask_value=0)(inputs)
 lstm1 = LSTM(DIM_LSTM1, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(msk)
 lstm2 = LSTM(DIM_LSTM2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(lstm1)
 
 att, pv = SoftAttention(lstm2)(lstm2)
 
 do1 = Dropout(0.1)(att)
 dense1 = Dense(DIM_DENSE1,activation='sigmoid')(do1)
 do2 = Dropout(0.1)(dense1)
 dense2 = Dense(DMI_DENSE2,activation='sigmoid')(do2)
 outputs = Dense(2,activation='softmax')(dense2)
 
 model=Model(outputs=outputs, inputs=inputs)
 model.compile(optimizer='adam', loss='binary_crossentropy')
 
 return model


## create directories for results and models
if not exists("./results/"):
 makedirs("./results/")

if not exists("./results/cv"):
 makedirs("./results/cv/")

if not exists("./weights/"):
 makedirs("./weights/")

if not exists("./weights/cv"):
 makedirs("./weights/cv/")


# parameters
MAX_LEN = 400 # maximum sequence length
DIM_ENC = 16 # dimension of a one-hot encoded vector (e.g., 4 (sequence) x 4 (structure) = 16)
DIM_LSTM1 = 20
DIM_LSTM2 = 10
DIM_DENSE1 = 400
DMI_DENSE2 = 100
N_EPOCH = 300
K_FOLD = 5
SPECIES = ['human','whole']



for _species in SPECIES:
 model = mymodel(MAX_LEN,DIM_ENC,DIM_LSTM1,DIM_LSTM2,DIM_DENSE1,DMI_DENSE2)
 W_init = model.get_weights() # initial model weights
 
 # test data (10%)
 #X_pos_test = import_data("./dataset/test/%s/%s_pos_test.fa" % (_species,_species))
 #X_neg_test = import_data("./dataset/test/%s/%s_neg_test.fa" % (_species,_species))
 #X_test = one_hot_wrap(X_pos_test + X_neg_test, MAX_LEN, DIM_ENC)
 #Y_test = to_categorical([0]*len(X_pos_test) + [1]*len(X_neg_test), num_classes = 2)
  
 WriteFile = open("./results/cv/%s_cv.rst" % _species ,"w")
 rst = []
 for fold in range(K_FOLD):
  # train
  X_pos_train = import_data("./dataset/cv/%s/train/%s_pos_train_f%d.fa" % (_species,_species,fold+1))
  X_neg_train = import_data("./dataset/cv/%s/train/%s_neg_train_f%d.fa" % (_species,_species,fold+1))
  X_train = one_hot_wrap(X_pos_train + X_neg_train, MAX_LEN, DIM_ENC)
  Y_train = to_categorical([0]*len(X_pos_train) + [1]*len(X_neg_train), num_classes = 2)
  
  # validation
  X_pos_val = import_data("./dataset/cv/%s/val/%s_pos_val_f%d.fa" % (_species,_species,fold+1))
  X_neg_val = import_data("./dataset/cv/%s/val/%s_neg_val_f%d.fa" % (_species,_species,fold+1))
  X_val = one_hot_wrap(X_pos_val + X_neg_val, MAX_LEN, DIM_ENC)
  Y_val = to_categorical([0]*len(X_pos_val) + [1]*len(X_neg_val), num_classes = 2)
  
  #model.set_weights(W_init)
  model.load_weights("./weights/cv/%s_f%d.hdf5" % (_species, fold+1))
  for i in range(N_EPOCH):
   print("fold:",str(fold+1),"epoch:",str(i+1))
   history = model.fit(X_train, Y_train, epochs=1, verbose=1, batch_size = 256, class_weight='auto')
   predictions = model.predict(X_val,verbose=0)
   rst = perfeval(predictions, Y_val, verbose=1)
   wrtrst(WriteFile,rst,fold,i)
  
  model.save_weights(filepath="./weights/cv/%s_f%d.hdf5" % (_species,fold+1), overwrite=True)
  
  #model.load_weights("./weights/recon/cv/%s_f%d.hdf5" % (_species,fold+1))
  #predictions = model.predict(X_val,verbose=0)
  #rst.append(perfeval(predictions, Y_val, verbose=1))
 
 WriteFile.close()



 
 #rst_avg = np.mean([f[:-1] for f in rst],axis=0)
 #model.load_weights("./weights/recon/cv/%s_f%d.hdf5" % (_species,2))
 #predictions = model.predict(X_test,verbose=0)
 #rst_test = perfeval(predictions, Y_test, verbose=1)









