# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:34:57 2022

@author: diviyah
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Bidirectional, Embedding
from tensorflow.keras.layers import Masking
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score



class ModelCreation():       
    def __init__(self):
        pass
    
    def final_model(self, output_node, vocab_size, embedding_dim = 64):
            
            model = Sequential()
            model.add(Input(shape=(333)))
            model.add(Embedding(vocab_size,embedding_dim))
            model.add(Bidirectional(LSTM(128,return_sequences=(True))))
            model.add(Dropout(0.2))
            model.add(Masking(mask_value=0)) #Masking Layer - Remove the 0 from padded data 
                                             # -> replace the 0 with the data values
            model.add(Bidirectional(LSTM(128,return_sequences=(True))))
            model.add(Dropout(0.2))
            model.add(LSTM(128))
            model.add(Dropout(0.2))
            model.add(Dense(128,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(output_node, activation='softmax'))           
            model.summary()
            
            return model


        
   
class ModelEvaluation():       
    def __init__(self):
        pass
    
    def eval_plot(self,hist):
        '''
        Generate graphs to evaluate model loss and accuracy 

        Parameters
        ----------
        hist : TYPE
            model fitted with Training and Validation data.

        Returns
        -------
        The plot of loss vs epoch and accuracy of epoch for train and validation dataset.

        '''
        hist.history.keys()

        plt.figure()
        plt.plot(hist.history['loss'],'r--', label='Training Loss')
        plt.plot(hist.history['val_loss'],label='Validation Loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'],'r--',label='Training acc')
        plt.plot(hist.history['val_acc'],label='Validation acc')
        plt.legend()
        plt.show()
        
        
    def model_eval(self,model,X_test,y_test,label):
        '''
        Generates confusion matrix and classification report based
        on predictions made by model using test dataset.

        Parameters
        ----------
        model : model
            Prediction model.
        x_test : ndarray
            Columns of test features.
        y_test : ndarray
            Target column of test dataset. 
        label : list
            Confusion matrix labels.

        Returns
        -------
        Returns numeric report of model.evaluate(), 
        classification report and confusion matrix.

        '''
        result = model.evaluate(X_test,y_test)
        
        print(result)
        
        y_true=np.argmax(y_test,axis=1)
        y_pred=np.argmax(model.predict(X_test),axis=1)
        
        print(y_true)
        print(y_pred)
        
        #Confusion_matrix
        cm=confusion_matrix(y_true,y_pred)
        print('Confussion matrix: \n ',cm)
        #show Confusion Matrix plot graph
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Reds)
        plt.show()
                
        #Classification report
        cr=classification_report(y_true,y_pred)
        print('Classification Report: \n ',cr)
        
        #Accuracy score
        acc=accuracy_score(y_true,y_pred)
        print('\n Accuracy score: \n ', acc)
       


    


