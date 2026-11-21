#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Identifying Algorithm

# In[1]:


#importing neccessary libraries
import numpy as np
import pandas as pd


# In[2]:


#putting it into a dataframe
# df = pd.DataFrame(pd.read_csv('spam.csv'))
df = pd.read_csv("spam.csv", encoding = "ISO-8859-1", engine = "python")


# In[3]:


df


# In[4]:


df.shape


# In[56]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# # 1.Data Cleaning

# In[5]:


df.info()


# In[96]:


# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[97]:


df.sample(5)


# In[79]:


# renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[80]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[81]:



df['target'] = encoder.fit_transform(df['target'])


# In[82]:


df.head()


# In[83]:


# missing values
data.isnull().sum()


# In[84]:


# check for duplicate values
data.duplicated().sum()


# In[92]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[93]:


df.duplicated().sum()


# In[94]:


df.shape


# # 2.EDA

# In[95]:


df.head()


# In[90]:


df['target'].value_counts()


# In[91]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[ ]:


# Data is imbalanced


# In[ ]:


import nltk


# In[ ]:


get_ipython().system('pip install nltk')


# In[ ]:


nltk.download('punkt')


# In[ ]:


df['num_characters'] = df['text'].apply(len)


# In[ ]:


df.head()


# In[ ]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


df.head()


# In[ ]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[ ]:


df.head()


# In[ ]:


df[['num_characters','num_words','num_sentences']].describe()


# In[ ]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[ ]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[ ]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[ ]:


sns.pairplot(df,hue='target')


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# # 3.Data Processing

# ## Lower case
# ## Tokenization
# ## Removing special characters
# ## Removing stop words and punctuation
# ## Stemming

# In[ ]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[ ]:


import re
import nltk
nltk.download('stopwords')


# In[ ]:


# import re

# nltk.download('stopwords')

# import nltk
# from nltk.corpus import stopwords
# #Create stopword list:
# stopwords = set(STOPWORDS)


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words('english'))


# In[ ]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[ ]:


df['text'][10]


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[ ]:


import string
df['transformed_text'] = df['text'].apply(transform_text)


# In[ ]:


df.head()


# In[ ]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[ ]:


conda install -c conda-forge wordcloud


# In[ ]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[ ]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[ ]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[ ]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[ ]:


df.head()


# In[ ]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[ ]:


len(spam_corpus)


# In[ ]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[ ]:


len(ham_corpus)


# In[ ]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Text Vectorization
# using Bag of Words
df.head()


# # 4. Model Building

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[ ]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[ ]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[ ]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[ ]:


X.shape


# In[ ]:


y = df['target'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[ ]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[ ]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[ ]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[ ]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:


# tfidf --> MNB


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[ ]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,

}


# In[ ]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[ ]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[ ]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[ ]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[ ]:


performance_df


# In[ ]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[ ]:


performance_df1


# In[ ]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[ ]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[ ]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[ ]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[ ]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[ ]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[ ]:



new_df_scaled.merge(temp_df,on='Algorithm')


# In[ ]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[ ]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[ ]:


voting.fit(X_train,y_train)


# In[ ]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[ ]:


from sklearn.ensemble import StackingClassifier


# In[ ]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[ ]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:








# In[ ]:





# In[ ]:





# # Identifying Fake news 

# In[21]:


import itertools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[22]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[16]:


data=pd.DataFrame(pd.read_csv('news.csv'))


# In[17]:


data


# # data inspection

# In[18]:


data.shape


# In[19]:


data.info()


# In[23]:


sns.heatmap(data.isnull(),cmap='magma',cbar=False)


# In[24]:


data.describe()


# In[25]:


data.head()
#print(data["class"])


# # Cleaning Data

# In[26]:


data.isnull().sum()*100/data.shape[0]


# In[27]:


#Preparing the data
# LIst of Variables to map 
varlist = ['label']

#defining the map function
def mar(x):
    return x.map({"REAL": 1, "FAKE": 0})

#then apply the mapping function to the variable
#just decided to map this way although we are just mapping one variable to the function
data['label']= mar(data['label'])
#.apply(mar)


# In[28]:


# checking the Fake News frame now
data.head()


# In[29]:


from sklearn.model_selection import train_test_split

# Removing the unwanted or unneccessary column

df = data.drop(['title','Unnamed: 0'], axis=1)


# In[30]:


df.head()


# In[31]:


# DEFINING A MANUAL FUNCTION TO REMOVE SPECIFIC STOP WORDS HERE

def drop_words(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
    


# In[32]:


# Applying the function to the data

df['text']=df['text'].apply(drop_words)


# In[33]:


df.head()


# In[34]:


# Moving on to define my Dependent and Independent Variables

x=df['text']
y=df['label']


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)


# In[36]:


# Time to vectorize the data
# Remember i imported the library of TfidVectorizer
vectorize= TfidfVectorizer(stop_words='english', max_df=0.7)
xv_train= vectorize.fit_transform(x_train)
xv_test= vectorize.transform(x_test)

#second coding line to also remove stop words
#making it the second time stop words have been filtered in the data set


# In[37]:


##Decision tree classifier


# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


DT= DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[40]:


DT.score(xv_test, y_test)


# In[41]:


predict_DT= DT.predict(xv_test)


# In[42]:


print(classification_report(y_test, predict_DT))


# In[ ]:





# In[ ]:





# In[43]:


##Gradient Boosting


# In[51]:


from sklearn.ensemble import GradientBoostingClassifier


# In[45]:


GBC=  GradientBoostingClassifier(random_state=0)


# In[46]:


GBC.fit(xv_train, y_train)


# In[47]:


GBC.score(xv_test, y_test)


# In[48]:


predict_GBC = GBC.predict(xv_test)


# In[50]:


print(classification_report(y_test, predict_GBC))


# In[ ]:




