# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:33:03 2022

@author: diviyah
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
import datetime
import pickle
import json
import os
import re
from module_article_NLP import ModelCreation,ModelEvaluation
# from tensorflow.keras.callbacks import EarlyStopping


#%%STATIC

CSV_URL='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','saved_model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(), 'model','saved_model_token.json')
OHE_PATH = os.path.join(os.getcwd(),'model','saved_ohe.pkl')

#%% Data Loading

df = pd.read_csv(CSV_URL)

df_copy=df.copy() #back up file

#%% Data Inspection

df.head(10)
df.tail(10)

df.info() # No NaNs
          # There are only categories and texts
          
df['category'].unique() # to get unique category variables
                        # tech, business,sport,entertainment,politics
                        

df['text'][0] # can see that an article has loads of words in them
df['category'][0] #tech

df['category'][100] #entertainment
df['text'][100]    #Negative reviews

# Is there anything there to remove? Funny Characters? Nope.
# if yes, need to remove because later under tokenization, it will consider them as a char

df.isna().sum() # no missing values

df.duplicated().sum() # We have 99 duplicated datas

sns.countplot(df.category) #business and sport are the mode for the category of the text

#%% STEP 3 - Data Cleaning

# To do - Remove duplicated datas
#       - Change all the alphabets into lower case

df = df.drop_duplicates() 
df.duplicated().sum()  # All the duplicated has been dropped


text= df['text'].values # Features: X #Extract the values to make text-category length
category = df['category'] # Target: category

for index,t in enumerate(text):
    text[index] = re.sub('.*?',' ',t)
    text[index] = re.sub('[^a-zA-Z]',' ',t).lower().split()


text[10] #all the words has been split into list of words now.

#%% Feature Selection
# No features to be selected for this case

#%% Data Preprocessing

# To do - Tokenization
#       - Padding and Truncating
#       - One Hot Encoding for Category
#       - Train Test Split

vocab_size = 15000
oov_token = 'OOV'

# =============================================================================
# # Tokenization
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text) #only the texts have to be tokenized
                               # categories --> OHE
word_index = tokenizer.word_index
print(word_index)            

# so need to encode all this into numbers to fit the review
train_sequences = tokenizer.texts_to_sequences(text)     
# so all the words now are in numerics 

print(train_sequences[10])
# =============================================================================
# =============================================================================
# # Padding & Truncatinng

# The number of words each article possesses
len(train_sequences[0]) #744 
len(train_sequences[1]) #296
len(train_sequences[2]) #247
len(train_sequences[3]) #345

# for padding we can choose either mean or median
length_of_text =[len(i) for i in train_sequences]

np.mean(length_of_text) # mean_words ==> 386
np.median(length_of_text) # median_words ==> 333
np.max(length_of_text) #max_words ==> 4469
np.min(length_of_text) #min_words ==> 90

# pick the reasonable padding value
# we are choosing median for our padding values
# Padding is to make each length to be similar
max_len = 333

padded_text= pad_sequences(train_sequences,
                              maxlen=max_len,
                              padding='post',
                              truncating='post')
                                # so now all in equal length already now
                                # 1 is OOV 
# =============================================================================
# =============================================================================
# One Hot Encoding - Category

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

# =============================================================================
# =============================================================================
# Train Test Split

X_train,X_test,y_train,y_test = train_test_split(padded_text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=7)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development

# np.shape(X_train)[1] = 333
nb_features=333
output_node=len(y_train[1])
nb_features=333

## for bidirectional - 
embedding_dim = 128

MC=ModelCreation()

model=MC.final_model(output_node, vocab_size)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

# early_stopping callbacks
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

hist=model.fit(X_train,y_train,
                validation_data=(X_test,y_test),
                batch_size=20,
                epochs=50,
                callbacks=[tensorboard_callback])

#%% Model Architecture
plot_model(model,show_shapes=True, show_layer_names=(True))           

#%% hist keys evaluation

ModelEvaluation().eval_plot(hist)   


#%% Model Evaluation reports

ME=ModelEvaluation()
ME.model_eval(model, X_test, y_test, label=['tech','business','sport',
                                            'entertainment','politics'])



#%% Model Saving
   
    
model.save(MODEL_SAVE_PATH)

# import tensorboard as tf
# tf.keras.callbacks.ModelCheckpoint(
#     MODEL_SAVE_PATH,
#     monitor='val_acc',
#     # verbose=0,
#     save_best_only=True,
#     save_weights_only=True,
#     mode='max',
#     # save_freq='epoch',
#     # options=None,
#     initial_value_threshold=None
# )


token_json = tokenizer.to_json()

with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

# tokenizer_sentiment is our dictionary now

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    

#%% DISCUSSION

# =============================================================================
# PROBLEM STATEMENT
# Provided text documents that has 5 categories, can we categorize unseen in articles into 5 categories as well?

# =============================================================================
# =============================================================================
# MODEL DEVELOPMENT AND EVALUATION
# =============================================================================
# Model accuracy by only using 2 LSTM layer has gotten us 27.8%
# Model accuracy using embedding layer increases to 31.2% accuracy
# Increasing the vocab_size increases the accuracy since the model getting smarter
# Adding masking layers spikes the accuracy by twice!!
# Model's accuracy increases again with an extra bidrectional layer
# And certainly increasing epoch size works the best as well!
# Decreasing the random_state helps the model to reach it's almost stability 
# Lastly, we have finalized the best model that is neither overfitted nor underfitted.

# =============================================================================
# =============================================================================
# # CHALLENGES
# =============================================================================
#  - model accuracy and loss' value were unstable eventhough run through the same code, due to its random_state.
#  - Most of the model built were over-fitted and model accuracy value decreases if early callbacks were added
#  - it takes a long for the computing process

# =============================================================================
# =============================================================================
# SUGGESTIONS
# - Need to stablize the model
# - Can do transfer learning and fine tune the model 
# Reference: https://www.tensorflow.org/guide/keras/transfer_learning
# =============================================================================
