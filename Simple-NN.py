#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential,Model
from keras.layers import Dense,LSTM, SpatialDropout1D, Embedding, Reshape, Concatenate
from keras.layers import Flatten
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


VOCAB_SIZE = 5000
OUTPUT_SIZE = 20
INPUT_SIZE = 60
HIDEN_SIZE = 20
IMAGE_DATA_SZ = 1024


# In[3]:


X_train = ["There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.", 
     "The cardiac, mediastinal and hilar contours are normal. Pulmonary vasculature is normal. Lungs are clear. No pleural effusion or pneumothorax is present. Multiple clips are again seen projecting over the left breast. Remote left-sided rib fractures are also re- demonstrated. Chexpert Prediction is No Finding",
    "Single frontal view of the chest provided. There is no focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal. Again seen are multiple clips projecting over the left breast and remote left-sided rib fractures. No free air below the right hemidiaphragm is seen.",
    "The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted. Chexpert Prediction is No Finding",
    "PA and lateral views of the chest provided. The lungs are adequately aerated. There is a focal consolidation at the left lung base adjacent to the lateral hemidiaphragm. There is mild vascular engorgement. There is bilateral apical pleural thickening. The cardiomediastinal silhouette is remarkable for aortic arch calcifications. The heart is top normal in size. Chexpert Prediction is Lung Opacity"]

Y_train = ["No acute cardiopulmonary process.", "No acute cardiopulmonary abnormality.","No acute intrathoracic process.","No acute cardiopulmonary process.","Focal consolidation at the left lung base, possibly representing aspiration or pneumonia. Central vascular engorgement."]


# In[4]:


df = pd.read_csv('./data/train-data.csv')


# In[5]:


df.head()


# In[6]:


# X_train = df['findings']


# In[7]:


# Y_train = df['impression']


# In[8]:


import gensim.downloader as api

model = api.load("glove-wiki-gigaword-300")


# In[9]:


import gensim
corpus = [gensim.utils.simple_preprocess(sentence) for sentence in X_train]


# In[10]:


#corpus


# In[11]:


w2v_model = gensim.models.Word2Vec(min_count=20,
                     window=2,
                     vector_size=100,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=4-1)


# In[12]:


model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=0)


# In[13]:


# Get the word vectors for your input sentences
train_vectors = []
embedding_size = 100
for sentence in corpus:
    sentence_vectors = []
    for word in sentence:
        if word in model.wv:
            sentence_vectors.append(tf.convert_to_tensor(np.array(model.wv[word])))
        else:
            # Handle out-of-vocabulary words
            sentence_vectors.append(tf.convert_to_tensor(np.array([0]*embedding_size)))
    while(len(sentence_vectors)<INPUT_SIZE):
        sentence_vectors.append(np.zeros(100))
    train_vectors.append(tf.convert_to_tensor(np.array(sentence_vectors)))


# In[14]:


train_vectors = np.array(train_vectors)


# In[15]:


test_corpus = [gensim.utils.simple_preprocess(sentence) for sentence in Y_train]


# In[16]:


# Get the word vectors for your input sentences
test_vectors = []
embedding_size = 100
for sentence in test_corpus:
    sentence_vectors = []
    for word in sentence:
        if word in model.wv:
            sentence_vectors.append(tf.convert_to_tensor(np.array(model.wv[word])))
        else:
            # Handle out-of-vocabulary words
            sentence_vectors.append(tf.convert_to_tensor(np.array([0]*embedding_size)))
    while(len(sentence_vectors)<OUTPUT_SIZE):
        sentence_vectors.append(np.zeros(100))
    test_vectors.append(tf.convert_to_tensor(np.array(sentence_vectors)))


# In[18]:


#test_vectors = np.array(test_vectors).reshape(len(test_vectors),1)


# In[19]:


#test_vectors = tf.convert_to_tensor(test_vectors)


# In[20]:


test_vectors[0].shape


# In[21]:


import numpy as np
temp = np.array(train_vectors)


# In[22]:


len(temp[4])


# In[23]:


len(corpus[1])


# In[24]:


import numpy as np
temp = np.array(test_vectors)


# In[25]:


len(temp[4])


# In[26]:


INPUT_SIZE = 6000


# In[27]:


# from keras.models import Sequential
# from keras.layers import Dense

# extra_data = np.zeros((5, IMAGE_DATA_SZ))

# # Define the model architecture
# model_nn = Sequential()
# model_nn.add(Flatten(input_shape=(60, 100)))
# model_nn.add(Dense(INPUT_SIZE//2, activation='relu', input_dim=INPUT_SIZE))
# model_nn.add(Dense(INPUT_SIZE//4, activation='relu'))

# # # Generate a random numpy array
# #random_array = np.random.rand(1024)

# # # Define a Lambda layer to concatenate the random array with the output of the second layer
# # concat_layer = Concatenate()([model_nn.layers[1].output, Lambda(lambda x: random_array)(model_nn.layers[1].output)])

# # # Add the concatenation layer to the model
# #model_nn.add(Dense(1024, activation='relu')(concat_layer))
# ## Add image 1024 data
# model_nn.add(Concatenate(axis=0))
# #model_nn.add(Dense(INPUT_SIZE//4+IMAGE_DATA_SZ, activation='relu'))
# model_nn.add(Dense(2000//4, activation='relu'))
# model_nn.add(Dense(2000//2, activation='relu'))
# model_nn.add(Dense(2000, activation='sigmoid'))
# model_nn.add(Reshape((20, 100)))


# # Compile the model
# model_nn.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# # Train the model on some data
# model_nn.fit(train_vectors, test_vectors, epochs=100, batch_size=5, validation_data=(train_vectors, test_vectors))

# # Evaluate the model on some test data
# loss, accuracy = model_nn.evaluate(train_vectors, test_vectors)

# # Make predictions using the model
# predictions = model_nn.predict(train_vectors)


# In[28]:


#len(predictions[0][0])


# In[29]:


#predictions[0].shape


# In[30]:


def glove_embeddings_to_sentences(glove_embeddings, model):
    # Find the closest words to each GloVe embedding vector
    sentences = []
    for glove_embedding in glove_embeddings:
        words = []
        for embedding in glove_embedding:
            print(model.wv.similar_by_vector(embedding, topn=1))
            try:
                words.append(model.wv.similar_by_vector(embedding, topn=1)[0][0])
            except:
                print(model.wv.similar_by_vector(embedding, topn=1))
                words.append(" ")

        # Convert the words to a sentence
        sentence = " ".join(words)
        sentences.append(sentence)

    return sentences


# In[31]:


#glove_embeddings_to_sentences(predictions, model)


# In[32]:


batch_size = 5


# In[33]:


# import numpy as np
# from keras.layers import Concatenate, Input

# input_shape = (60, 100)

# # create a numpy array of size (batch_size, 1024)
# extra_data = np.zeros((batch_size, 1024))
# extra_data_shape = (1024,)

# input_layer = Input(shape=input_shape)

# x = Flatten()(input_layer)
# x = Dense(INPUT_SIZE//2, activation='relu', input_dim=INPUT_SIZE)(x)
# x = Dense(INPUT_SIZE//4, activation='relu')(x)
# # create the extra data input layer
# extra_data_layer = Input(shape=extra_data_shape)

# x = Concatenate()([x, extra_data_layer])
# x = Dense(2000//4, activation='relu')(x)
# x = Dense(2000//2, activation='relu')(x)
# x = Dense(2000, activation='sigmoid')(x)
# output_layer = Reshape((20, 100))(x)
# model_nn = Model(inputs=[input_layer, extra_data_layer], outputs=output_layer)


# # model_nn = Sequential()
# # model_nn.add(Flatten(input_shape=(60, 100)))
# # model_nn.add(Dense(INPUT_SIZE//2, activation='relu', input_dim=INPUT_SIZE))
# # model_nn.add(Dense(INPUT_SIZE//4, activation='relu'))

# # # concatenate the extra data array
# # model_nn.add(Concatenate(axis=1))

# # # add another layer
# # model_nn.add(Dense(128, activation='relu'))
# # model_nn.add(Dense(10, activation='softmax'))


# In[34]:


# Compile the model
#model_nn.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[35]:


#test_vectors.shape


# In[36]:


# Train the model on some data
# model_nn.fit([train_vectors,np.zeros((batch_size, 1024))], test_vectors, epochs=100, batch_size=5, validation_data=(train_vectors, test_vectors))


# In[37]:


import numpy as np
from keras.layers import Concatenate, Input, Dense, Flatten, Reshape
from keras.models import Model

# define the input shapes
input_shape = (60, 100)
extra_data_shape = (1024,)

# create a numpy array of size (batch_size, 1024)
batch_size = 5
extra_data = np.zeros((batch_size, 1024))

# create the input layers
input_layer = Input(shape=input_shape)
extra_data_layer = Input(shape=extra_data_shape)

# add the layers to the model
x = Flatten()(input_layer)
x = Dense(INPUT_SIZE//3, activation='relu', input_dim=INPUT_SIZE)(x)
x = Dense(INPUT_SIZE//9, activation='relu')(x)
x = Dense(INPUT_SIZE//27, activation='relu')(x)
x = Dense(INPUT_SIZE//81, activation='relu')(x)
# x = Dense(INPUT_SIZE//243, activation='relu')(x)
# x = Dense(INPUT_SIZE//729, activation='relu')(x)



# concatenate the extra data with the output of the last layer
x = Concatenate()([x, extra_data_layer])

# x = Dense(2000//729, activation='relu')(x)
# x = Dense(2000//243, activation='relu')(x)
x = Dense(2000//81, activation='relu')(x)
x = Dense(2000//27, activation='relu')(x)
x = Dense(2000//9, activation='relu')(x)
x = Dense(2000//3, activation='sigmoid')(x)
x = Dense(2000, activation='sigmoid')(x)
output_layer = Reshape((20, 100))(x)

# create the model
model_nn = Model(inputs=[input_layer, extra_data_layer], outputs=output_layer)

# compile the model
model_nn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# create some dummy input data
input_data = np.random.rand(batch_size, 60, 100)

# train the model for 10 epochs
epochs = 10
for i in range(epochs):
    # create some dummy target data
    target_data = np.random.randint(2, size=(batch_size, 20, 100))
    # train the model for one epoch
    model_nn.fit([train_vectors, extra_data], target_data, batch_size=batch_size)


# In[ ]:





# In[ ]:




