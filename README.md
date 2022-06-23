![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

# NLP_Article_Classification_Problem_RNN_Bidirectional_Model
 Categorized news articles into 5 categories ( Sport, Tech, Business, Entertainment and  Politics) by using Recurrent Neural Network method to develop the model.

## PROBLEM STATEMENT
### Provided text documents that has 5 categories, can we categorize unseen in articles into 5 categories as well?

## Questions:
###     1. What kind of data are we dealing with?
        - There are only 2 variables namely text(features) and category (target).
        
 ![plt](https://user-images.githubusercontent.com/105897390/175362626-f4c2d6c1-81a2-4c99-ab17-b299b1fc8270.png)
 
*The bar chart of the data*

###     2. Do we have missing values?
        - None
        
###     3. Do we have duplicated datas?
        - There were 99 duplicated datas 
        
###     4. Do we have extreme values?
        - None due to the nature of the data.
       
###     5. How to choose the features to make the best out of the provided data?
        - We dont have to choose features. As all the texts from the articles are naturally considered as features. 


## MODEL DEVELOPMENT AND EVALUATION
- Model accuracy by only using 2 LSTM layer has gotten us 27.8%
- Model accuracy using embedding layer increases to 31.2% accuracy
- Increasing the vocab_size increases the accuracy since the model getting smarter
- Adding masking layers spikes the accuracy by twice!!
- Model's accuracy increases again with an extra bidrectional layer
- And certainly increasing epoch size works the best as well!
- Decreasing the random_state helps the model to reach it's almost stability 
- Lastly, we have finalized the model that is neither overfitted nor underfitted.

Attached below are two images sniped from launched TensorBoard

![tensorboard_alot](https://user-images.githubusercontent.com/105897390/175363330-9f224053-0ee2-43ce-abe5-cb32c40d3ecb.png)

*The multiple line graphs is one of the indication that we had to redevelop the model multiple times due to its unstability and that model often overfits. Early callbacks didnt help to increase the accuracy too.*

![tensorboard_final](https://user-images.githubusercontent.com/105897390/175363829-d22c54d8-8140-472d-9d02-1b7cdd6a623f.png)

*This is the model chosen that doesnt overfits and reached better accuracy than the others.*


![acc_0 75](https://user-images.githubusercontent.com/105897390/175364043-db4c50e4-9b20-4a1e-9088-7c67be182255.png)



![loss_0 75](https://user-images.githubusercontent.com/105897390/175364058-120a355c-a5eb-452a-95cd-bc8b014aa446.png)


*The last 2 plots are the plots of accuracy and loss value of both training and test dataset against epoch set for the RNN model.*


![model_eval_0 75](https://user-images.githubusercontent.com/105897390/175364244-6c656951-01d8-486a-a631-0ee7f33c55d0.png)
*This is the display of our model accuracy - 75% with f1-score as 0.76.*


![model](https://user-images.githubusercontent.com/105897390/175364512-6bfeea7e-e160-43d2-b1a2-c5be6f72657b.png)

*Last but not least, the image depicts out model's architecture.*
