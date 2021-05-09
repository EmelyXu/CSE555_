import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import GridSearchCV
import numpy as np
from tensorflow import keras
import argparse
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, load_model
import keras.backend as K
from sklearn.model_selection import train_test_split
from data_helpers import Dataloader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import os, pickle
import numpy as np

mport numpy as np
import pandas as pd
import pickle
import os, sys
from collections import Counter, defaultdict

###################################################################################################################################

# Hyperparams
max_length=50 # Maximum length of the sentence

class Dataloader:
    
    def __init__(self, mode=None):

        try:
            assert(mode is not None)
        except AssertionError as e:
            print("Set mode as 'Sentiment' or 'Emotion'")
            exit()

        self.MODE = mode # Sentiment or Emotion classification mode
        self.max_l = max_length

        """
            Loading the dataset:
                - revs is a dictionary with keys/value:
                    - text: original sentence
                    - split: train/val/test :: denotes the which split the tuple belongs to
                    - y: label of the sentence
                    - dialog: ID of the dialog the utterance belongs to
                    - utterance: utterance number of the dialog ID
                    - num_words: number of words in the utterance
                - W: glove embedding matrix
                - vocab: the vocabulary of the dataset
                - word_idx_map: mapping of each word from vocab to its index in W
                - label_index: mapping of each label (emotion or sentiment) to its assigned index, eg. label_index['neutral']=0
        """
        x = pickle.load(open("./data/pickles/data_{}.p".format(self.MODE.lower()),"rb"))
        revs, self.W, self.word_idx_map, self.vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        self.num_classes = len(label_index)
        print("Labels used for this classification: ", label_index)


        # Preparing data
        self.train_data, self.val_data, self.test_data = {},{},{}
        for i in range(len(revs)):
            
            utterance_id = revs[i]['dialog']+"_"+revs[i]['utterance']
            sentence_word_indices = self.get_word_indices(revs[i]['text'])
            label = label_index[revs[i]['y']]

            if revs[i]['split']=="train":
                self.train_data[utterance_id]=(sentence_word_indices,label)
            elif revs[i]['split']=="val":
                self.val_data[utterance_id]=(sentence_word_indices,label)
            elif revs[i]['split']=="test":
                self.test_data[utterance_id]=(sentence_word_indices,label)


        # Creating dialogue:[utterance_1, utterance_2, ...] ids
        self.train_dialogue_ids = self.get_dialogue_ids(self.train_data.keys())
        self.val_dialogue_ids = self.get_dialogue_ids(self.val_data.keys())
        self.test_dialogue_ids = self.get_dialogue_ids(self.test_data.keys())

        # Max utternance in a dialog in the dataset
        self.max_utts = self.get_max_utts(self.train_dialogue_ids, self.val_dialogue_ids, self.test_dialogue_ids)


    def get_word_indices(self, data_x):
        length = len(data_x.split())
        return np.array([self.word_idx_map[word] for word in data_x.split()] + [0]*(self.max_l-length))[:self.max_l]

    def get_dialogue_ids(self, keys):
        ids=defaultdict(list)
        for key in keys:
            ids[key.split("_")[0]].append(int(key.split("_")[1]))
        for ID, utts in ids.items():
            ids[ID]=[str(utt) for utt in sorted(utts)]
        return ids

    def get_max_utts(self, train_ids, val_ids, test_ids):
        max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
        max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
        max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
        return np.max([max_utts_train, max_utts_val, max_utts_test])

    def get_one_hot(self, label):
        label_arr = [0]*self.num_classes
        label_arr[label]=1
        return label_arr[:]


    def get_dialogue_audio_embs(self):
        key = list(self.train_audio_emb.keys())[0]
        pad = [0]*len(self.train_audio_emb[key])

        def get_emb(dialogue_id, audio_emb):
            dialogue_audio=[]
            for vid in dialogue_id.keys():
                local_audio=[]
                for utt in dialogue_id[vid]:
                    try:
                        local_audio.append(audio_emb[vid+"_"+str(utt)][:])
                    except:
                        print(vid+"_"+str(utt))
                        local_audio.append(pad[:])
                for _ in range(self.max_utts-len(local_audio)):
                    local_audio.append(pad[:])
                dialogue_audio.append(local_audio[:self.max_utts])
            return np.array(dialogue_audio)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_audio_emb)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_audio_emb)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_audio_emb)

    def get_dialogue_text_embs(self):
        key = list(self.train_data.keys())[0]
        pad = [0]*len(self.train_data[key][0])

        def get_emb(dialogue_id, local_data):
            dialogue_text = []
            for vid in dialogue_id.keys():
                local_text = []
                for utt in dialogue_id[vid]:
                    local_text.append(local_data[vid+"_"+str(utt)][0][:])
                for _ in range(self.max_utts-len(local_text)):
                    local_text.append(pad[:])
                dialogue_text.append(local_text[:self.max_utts])
            return np.array(dialogue_text)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_data)


    def get_dialogue_labels(self):

        def get_labels(ids, data):
            dialogue_label=[]

            for vid, utts in ids.items():
                local_labels=[]
                for utt in utts:
                    local_labels.append(self.get_one_hot(data[vid+"_"+str(utt)][1]))
                for _ in range(self.max_utts-len(local_labels)):
                    local_labels.append(self.get_one_hot(1)) # Dummy label
                dialogue_label.append(local_labels[:self.max_utts])
            return np.array(dialogue_label)

        self.train_dialogue_label=get_labels(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_label=get_labels(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_label=get_labels(self.test_dialogue_ids, self.test_data)

    def get_dialogue_lengths(self):

        self.train_dialogue_length, self.val_dialogue_length, self.test_dialogue_length=[], [], []
        for vid, utts in self.train_dialogue_ids.items():
            self.train_dialogue_length.append(len(utts))
        for vid, utts in self.val_dialogue_ids.items():
            self.val_dialogue_length.append(len(utts))
        for vid, utts in self.test_dialogue_ids.items():
            self.test_dialogue_length.append(len(utts))

    def get_masks(self):

        self.train_mask = np.zeros((len(self.train_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.train_dialogue_length)):
            self.train_mask[i,:self.train_dialogue_length[i]]=1.0
        self.val_mask = np.zeros((len(self.val_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.val_dialogue_length)):
            self.val_mask[i,:self.val_dialogue_length[i]]=1.0
        self.test_mask = np.zeros((len(self.test_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.test_dialogue_length)):
            self.test_mask[i,:self.test_dialogue_length[i]]=1.0


    def load_audio_data(self, ):

        AUDIO_PATH = "./data/pickles/audio_embeddings_feature_selection_{}.pkl".format(self.MODE.lower())
        self.train_audio_emb, self.val_audio_emb, self.test_audio_emb = pickle.load(open(AUDIO_PATH,"rb"))
        
        self.get_dialogue_audio_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()

    def load_text_data(self, ):

        self.get_dialogue_text_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()


    def load_bimodal_data(self,):
        
        TEXT_UNIMODAL = "./data/pickles/text_{}.pkl".format(self.MODE.lower())
        AUDIO_UNIMODAL = "./data/pickles/audio_{}.pkl".format(self.MODE.lower())

        #Load features
        train_text_x, val_text_x, test_text_x = pickle.load(open(TEXT_UNIMODAL, "rb"), encoding='latin1')
        train_audio_x, val_audio_x, test_audio_x = pickle.load(open(AUDIO_UNIMODAL, "rb"), encoding='latin1')

        def concatenate_fusion(ID, text, audio):
            bimodal=[]
            for vid, utts in ID.items():
                bimodal.append(np.concatenate( (text[vid],audio[vid]) , axis=1))
            return np.array(bimodal)

        self.train_dialogue_features = concatenate_fusion(self.train_dialogue_ids, train_text_x, train_audio_x)
        self.val_dialogue_features = concatenate_fusion(self.val_dialogue_ids, val_text_x, val_audio_x)
        self.test_dialogue_features = concatenate_fusion(self.test_dialogue_ids, test_text_x, test_audio_x)

        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()




class bc_LSTM:

    def __init__(self):
        self.classification_mode = "Sentiment"
        self.modality = "bimodal"
        self.PATH = "./data/models/bimodal_weights_sentiment.hdf5"
        self.OUTPUT_PATH = "./data/pickles/bimodal_sentiment.pkl"
        print("Model initiated for bimodal classification")


    def load_data(self,):

        print('Loading data')
        self.data = Dataloader(mode = self.classification_mode)
        self.data.load_bimodal_data()
        self.train_x = self.data.train_dialogue_features
        self.val_x = self.data.val_dialogue_features
        self.test_x = self.data.test_dialogue_features

        self.train_y = self.data.train_dialogue_label
        self.val_y = self.data.val_dialogue_label
        self.test_y = self.data.test_dialogue_label

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

        self.train_id = self.data.train_dialogue_ids.keys()
        self.val_id = self.data.val_dialogue_ids.keys()
        self.test_id = self.data.test_dialogue_ids.keys()

        self.sequence_length = self.train_x.shape[1]
        
        self.classes = self.train_y.shape[2]
            


    def calc_test_result(self, pred_label, test_label, test_mask):

        true_label=[]
        predicted_label=[]

        for i in range(pred_label.shape[0]):
            for j in range(pred_label.shape[1]):
                if test_mask[i,j]==1:
                    true_label.append(np.argmax(test_label[i,j] ))
                    predicted_label.append(np.argmax(pred_label[i,j] ))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=4))
        print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

        
        

    def get_bimodal_model(self):

        # Modality specific hyperparameters
        self.epochs = 10
        self.batch_size = 10

        # Modality specific parameters
        self.embedding_dim = self.train_x.shape[2]

        print("Creating Model...")
        
        inputs = Input(shape=(self.sequence_length, self.embedding_dim), dtype='float32')
        masked = Masking(mask_value =0)(inputs)
        lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.4), name="utter")(masked)
        output = TimeDistributed(Dense(self.classes,activation='softmax'))(lstm)

        model = Model(inputs, output)
        return model




    def train_model(self):

        checkpoint = ModelCheckpoint(self.PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        model = self.get_bimodal_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(self.train_x, self.train_y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        sample_weight=self.train_mask,
                        shuffle=True,
                        callbacks=[early_stopping, checkpoint],
                        validation_data=(self.val_x, self.val_y, self.val_mask))

        self.test_model()



    def test_model(self):

        model = load_model(self.PATH)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("utter").output)

        intermediate_output_train = intermediate_layer_model.predict(self.train_x)
        intermediate_output_val = intermediate_layer_model.predict(self.val_x)
        intermediate_output_test = intermediate_layer_model.predict(self.test_x)

        train_emb, val_emb, test_emb = {}, {}, {}
        for idx, ID in enumerate(self.train_id):
            train_emb[ID] = intermediate_output_train[idx]
        for idx, ID in enumerate(self.val_id):
            val_emb[ID] = intermediate_output_val[idx]
        for idx, ID in enumerate(self.test_id):
            test_emb[ID] = intermediate_output_test[idx]
        pickle.dump([train_emb, val_emb, test_emb], open(self.OUTPUT_PATH, "wb"))

        self.calc_test_result(model.predict(self.test_x), self.test_y, self.test_mask)
        

model_ = bc_LSTM()
model_.load_data()
model_.train_model()

class Sentiment:
    Ngative = "negative"
    Neutral = "neutral"
    Postive = "positive"

class Emotion:
    Joy = "joy"
    Surprise = "surprise"
    Neutral = "neutral"
    Fear = "fear"
    Sadness = "sadness"
    Disgust = "disgust"
    Anger = "anger"

class Dialogue:
    def __init__(self, text, u_id, d_id, speaker, sentiment, emotion):
        self.text = text
        self.u_id = u_id
        self.sentiment = sentiment
        self.d_id = d_id
        self.speaker = speaker
        self.emotion = emotion
        
class DialogueContainer:
    def __init__(self, dialogues):
        self.dialogues = dialogues
        
    def get_text(self):
        return [x.text for x in self.dialogues]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.dialogues]
    
    def get_emotion(self):
        return [x.emotion for x in self.dialogues]
    
    def evenly_distribute(self):
        neutral = list(filter(lambda x: x.sentiment == Sentiment.Neutral, self.dialogues))
        negative = list(filter(lambda x: x.sentiment == Sentiment.Ngative, self.dialogues))
        positive = list(filter(lambda x: x.sentiment == Sentiment.Postive, self.dialogues))
        neutral_shrunk = positive[:len(negative)]
        #self.dialogues = negative + neutral_shrunk
        random.shuffle(self.dialogues)
        
        
file_name = './data/test_sent_emo.json'

dialogues = []
with open(file_name) as f:
    for line in f:
        dialogue = json.loads(line)
        dialogues.append(Dialogue(dialogue['Utterance'], dialogue['Utterance_ID'],dialogue['Dialogue_ID'],dialogue['Speaker'],dialogue['Sentiment'],dialogue['Emotion']))
        
test = dialogues
test[1].text

file_name = './data/train_sent_emo.json'

dialogues = []
with open(file_name) as f:
    for line in f:
        dialogue = json.loads(line)
        dialogues.append(Dialogue(dialogue['Utterance'], dialogue['Utterance_ID'],dialogue['Dialogue_ID'],dialogue['Speaker'],dialogue['Sentiment'],dialogue['Emotion']))
training = dialogues
training[1].emotion


train_container = DialogueContainer(training)
test_container = DialogueContainer(test)


train_x = train_container.get_text()
train_y = train_container.get_sentiment()
train_z = train_container.get_emotion()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()
test_z = test_container.get_emotion()

print("------ Samples of Sentiment ----")
print("Number of total positive samples: "+str(train_y.count(Sentiment.Postive)))
print("Number of total negative samples: "+str(train_y.count(Sentiment.Ngative)))
print("Number of total neutral samples: "+str(train_y.count(Sentiment.Neutral)))
print(" ")
print("------ Samples of Emotion ------")
print("Number of total joy samples: "+ str(train_z.count(Emotion.Joy)))
print("Number of total surprise samples: "+ str(train_z.count(Emotion.Surprise)))
print("Number of total neutral samples: "+ str(train_z.count(Emotion.Neutral)))
print("Number of total fear samples: "+ str(train_z.count(Emotion.Fear)))
print("Number of total sadness samples: "+ str(train_z.count(Emotion.Sadness)))
print("Number of total disgust samples: "+ str(train_z.count(Emotion.Disgust)))
print("Number of total anger samples: "+ str(train_z.count(Emotion.Anger)))


vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

print(train_x[0])
#print(train_x_vectors[0])


clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)
print("Result of linear")

print("Correctness of navy bayes: "+str(clf_svm.score(test_x_vectors, test_y)))

#clf_svm.predict(test_x_vectors[0])


clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)
clf_gnb.fit(train_x_vectors, train_y)
print("Result of navy bayes")

print("Correctness of navy bayes: "+str(clf_gnb.score(test_x_vectors, test_y)))

#clf_gnb.predict(test_x_vectors[0])

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

print("Correctness of navy bayes: "+str(clf_log.score(test_x_vectors, test_y)))


nega = os.path.join('./data/rps/negative')
neut = os.path.join('./data/rps/neutral')
posi = os.path.join('./data/rps/positive')

print('total training negative emotion images:', len(os.listdir(nega)))
print('total training neural emotion images:', len(os.listdir(neut)))
print('total training positive emotion images:', len(os.listdir(posi)))

nega_files = os.listdir(nega)
neut_files = os.listdir(neut)
posi_files = os.listdir(posi)


%matplotlib inline

pic_index = 3



next_nega = [os.path.join(nega, fname)
                for fname in nega_files[pic_index-1:pic_index]]
next_neut = [os.path.join(neut, fname)
                for fname in neut_files[pic_index-1:pic_index]]
next_posi = [os.path.join(posi, fname)
                for fname in posi_files[pic_index-1:pic_index]]

print("Some examples of the train images:")
for i, img_path in enumerate(next_nega + next_neut + next_posi):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./data/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=5,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "./data/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=15, validation_data = validation_generator, verbose = 1)

model.save("rps.h5")

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

import numpy as np
import os

from keras.preprocessing import image

path = './data/test_images/nt10.png'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
#print(fn)

print("neg net pos")
print(classes)


img_ = mpimg.imread(path)
plt.imshow(img_)
plt.axis('Off')
plt.show()
