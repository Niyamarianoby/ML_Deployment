#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
seed=42 # 1 and 42 give good result for random state;0=False


# In[2]:


# Read original dataset
iris_df = pd.read_csv('iris.csv')
iris_df.sample(frac=1, random_state=seed)


# In[3]:


# selecting features and target data
X = iris_df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = iris_df[['Species']]


# In[4]:


# split data into train and test sets
# 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# In[19]:


X_train


# In[5]:


# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)


# In[6]:


# train the classifier on the training data
clf.fit(X_train, y_train)


# In[7]:


# predict on the test set
y_pred = clf.predict(X_test)


# In[8]:


# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  


# In[9]:


# save the model to disk
pickle.dump(clf, open(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk1",'wb'))


# In[12]:


import pickle
with open(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk1", 'rb') as f:
    model2 = pickle.load(f)   


def predict_(data, model = model2):   
    return model.predict(data)


# In[39]:


import pandas as pd
import numpy as np
from tkinter import *
from sklearn.ensemble import RandomForestClassifier

df = pd.read_pickle(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk1")

root = Tk()
root.title("IRIS Flower Prediction")
root.geometry('500x400')

Label(root, text="Sepal_length", font="Times 15").grid(row=0, column=0, padx=50, pady=10)
Label(root, text="Sepal_width", font="Times 15").grid(row=1, column=0, pady=10)
Label(root, text="Petal_length", font="Times 15").grid(row=2, column=0, pady=10)
Label(root, text="Petal_width", font="Times 15").grid(row=3, column=0, pady=10)
Label(root, text="Prediction", font="Times 20").grid(row=5, column=0, pady=10)

textbox = Text(root, height=3, width=20, )
textbox.grid(row=5, column=1)

input_text = StringVar()
input_text1 = StringVar()
input_text2 = StringVar()
input_text3 = StringVar()
result = StringVar()

e1 = Entry(root, font=1, textvariable=input_text)
e1.grid(row=0, column=1)
e2 = Entry(root, font=1, textvariable=input_text1)
e2.grid(row=1, column=1)
e3 = Entry(root, font=1, textvariable=input_text2)
e3.grid(row=2, column=1)
e4 = Entry(root, font=1, textvariable=input_text3)
e4.grid(row=3, column=1)


def entryClear():  #can done in another method
    input_text.set("")
    input_text1.set("")
    input_text2.set("")
    input_text3.set("")
    result.set("")
    textbox.delete(1.0, END)


def getpredict():
    lst = [float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get())]
    eg = np.array(lst)
    eg = eg.reshape(1, -1) #-1 reads as num of elements present...convert to 2D array

    x = iris_df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
    y = iris_df[['Species']]
    

    model = RandomForestClassifier()
    model.fit(x, y)

    predict = model.predict(eg)
    textbox.insert(END, predict)


Button(root, text='Clear', font=10, width=8, bg="red", fg="white", command=entryClear).grid(row=4, column=0,
                                                                                                pady=10)
Button(root, text='Predict', font=10, width=8, bg="pink", fg="white", command=getpredict).grid(row=4, column=1,
                                                                                                  pady=10)

root.mainloop()


# In[16]:


import numpy as np
a=np.array([1,2,3])
c=a.reshape(1,-1)


# In[17]:


c


# In[21]:


#dataset2


# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
seed=42


# In[2]:


Fish = pd.read_csv('Fish.csv')
Fish.sample(frac=1, random_state=seed)


# In[3]:


X = Fish[['Weight', 'Length1', 'Length2','Length3','Height', 'Width']]
y = Fish[['Species']]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# In[9]:


clf = RandomForestClassifier(n_estimators=100)


# In[10]:


clf.fit(X_train, y_train)


# In[11]:


y_pred = clf.predict(X_test)


# In[12]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  


# In[13]:


pickle.dump(clf, open(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk2",'wb'))


# In[14]:


import pickle
with open(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk2", 'rb') as f:
    m = pickle.load(f)   


def predict_(data, model = m):   
    return model.predict(data)


# In[16]:


import pandas as pd
import numpy as np
from tkinter import *
from sklearn.ensemble import RandomForestClassifier

df = pd.read_pickle(r"C:\Users\niyam\OneDrive\Desktop\rf_model.pk2")

root = Tk()
root.title("Fish Prediction")
root.geometry('500x500')

Label(root, text="Weight", font="Times 15").grid(row=0, column=0, padx=50, pady=10)
Label(root, text="Length1", font="Times 15").grid(row=1, column=0, pady=10)
Label(root, text="Length2", font="Times 15").grid(row=2, column=0, pady=10)
Label(root, text="Length3", font="Times 15").grid(row=3, column=0, pady=10)
Label(root, text="Height", font="Times 15").grid(row=4, column=0, pady=10)
Label(root, text="Width", font="Times 15").grid(row=5, column=0, pady=10)
Label(root, text="Prediction", font="Times 20").grid(row=8, column=0, pady=10)

textbox = Text(root, height=3, width=20, )
textbox.grid(row=9, column=1)

input_text = StringVar()
input_text1 = StringVar()
input_text2 = StringVar()
input_text3 = StringVar()
input_text4 = StringVar()
input_text5 = StringVar()
result = StringVar()

e1 = Entry(root, font=1, textvariable=input_text)
e1.grid(row=0, column=1)
e2 = Entry(root, font=1, textvariable=input_text1)
e2.grid(row=1, column=1)
e3 = Entry(root, font=1, textvariable=input_text2)
e3.grid(row=2, column=1)
e4 = Entry(root, font=1, textvariable=input_text3)
e4.grid(row=3, column=1)
e5 = Entry(root, font=1, textvariable=input_text4)
e5.grid(row=4, column=1)
e6 = Entry(root, font=1, textvariable=input_text5)
e6.grid(row=5, column=1)


def entryClear():  #can done in another method
    input_text.set("")
    input_text1.set("")
    input_text2.set("")
    input_text3.set("")
    input_text4.set("")
    input_text5.set("")
    result.set("")
    textbox.delete(1.0, END)


def getpredict():
    lst = [float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get()),float(e5.get()),float(e6.get())]
    eg = np.array(lst)
    eg = eg.reshape(1, -1) #-1 reads as num of elements present...convert to 2D array

    x = Fish[['Weight', 'Length1', 'Length2','Length3', 'Width','Height']]
    y = Fish[['Species']]
    

    model = RandomForestClassifier()
    model.fit(x, y)

    predict = model.predict(eg)
    textbox.insert(END, predict)


Button(root, text='Clear', font=10, width=8, bg="red", fg="white", command=entryClear).grid(row=7, column=0,
                                                                                                pady=10)
Button(root, text='Predict', font=10, width=8, bg="pink", fg="white", command=getpredict).grid(row=7, column=1,
                                                                                                  pady=10)

root.mainloop()


# In[ ]:




