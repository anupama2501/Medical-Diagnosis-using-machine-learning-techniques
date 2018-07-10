# Importing the libraries
import numpy as np
import pandas as pd
import re 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# Importing the Training dataset
dataset_train = pd.read_csv('train.dat' , header=None, delimiter= '\t' , quoting=3)
dataset_train.columns = ['Classification' , 'Diagnosis']

#Importing the Test Dataset
dataset_test = pd.read_csv('test.dat' , header=None, delimiter= '\t' , quoting=3)
dataset_test.columns = ['Diagnosis']

#Cleaning the text for the training data

corpus_train  = []
len_dataset_train = len(dataset_train['Diagnosis'])

for i in range(0,len_dataset_train):

    review_train = re.sub('[^a-zA-Z0-9%]' ,' ' , dataset_train['Diagnosis'][i])
    review_train = review_train.lower()
    review_train = review_train.split()
    review_train = [wnl.lemmatize(word) for word in review_train if not word in set(stopwords.words('english'))]
    review_train =' '.join(review_train)
    corpus_train.append(review_train)
    if( (i+1)%1000 == 0 ):
        print( "Processing Record: %d of %d\n" % ( i+1, len_dataset_train ) )                                                                   


#Creating the bag of words for the training data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 3000)
train_data_features = cv.fit_transform(corpus_train).toarray()
y= dataset_train.iloc[:,0]

#Running TF IDF for Term Frequency for training data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_data_features = tfidf_transformer.fit_transform(train_data_features)
train_data_features.shape


#Cleaning the text for test data 

corpus_test  = []
len_dataset_test = len(dataset_test['Diagnosis'])

for i in range(0,len_dataset_test):
    review_test = re.sub('[^a-zA-Z0-9%]' ,' ' , dataset_test['Diagnosis'][i])
    review_test = review_test.lower()
    review_test = review_test.split()
    review_test = [wnl.lemmatize(word) for word in review_test if not word in set(stopwords.words('english'))]
    review_test =' '.join(review_test)
    corpus_test.append(review_test)
    if( (i+1)%1000 == 0 ):
        print( "Processing Record: %d of %d\n" % ( i+1, len_dataset_test) )

#Creating the bag of words for the test data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 3000)
test_data_features= cv.fit_transform(corpus_test).toarray()
print( "Finished creating the bag of words for training data")

#Running TF IDF for Term Frequency for Test data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
test_data_features = tfidf_transformer.fit_transform(test_data_features)
test_data_features.shape
print( "Finished calculating the term frequency for training data")


#Classification Technique

print( "Creating a classification model, and predicing the results ")

# Splitting the Training data into a train-test split to check accuracy before running on actual data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_features, y, test_size = 0.25, random_state = 0)

# Fitting SGDC to the Training set

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
classifier.fit(X_train, y_train)

# Predicting the Training set results
y_pred = classifier.predict(X_test)

#Checking F1 Score
from sklearn.metrics import f1_score

print(f1_score(y_test, y_pred, average='macro'))  
print(f1_score(y_test, y_pred, average='micro') ) 
print(f1_score(y_test, y_pred, average='weighted'))  
f1_score(y_test, y_pred, average=None)


#Fitting the model with the complete training data, and running a prediction on the test data

y_pred_test = classifier.predict(test_data_features)

# Copy the results to a pandas dataframe
output = pd.DataFrame( data=y_pred_test )

# Use pandas to write the comma-separated output file
output.to_csv( "Anupama_011325041_Output.dat", header=None, index=False, quoting=3 )
print( "Created Anupama_011325041_Output.dat with the predicted results")



