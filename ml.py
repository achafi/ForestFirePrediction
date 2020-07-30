import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')

# Split data into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# train the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
#xt = np.array([[50,15,30]])
#print(log_reg.predict(xt))

#Dump the model into pickle file
pickle.dump(log_reg,open('model.pkl','wb'))
