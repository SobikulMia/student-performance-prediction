
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# import seaborn as sns
# Load for data
data = pd.read_excel("StudentsPerformance.xlsx")
print(data.isnull().sum())
print(data.info())
print(data.head())         
# Encode catagoricl data
data_encoding = pd.get_dummies(data,drop_first=True)
# Feature and target data
x = data_encoding.drop(["math score", "reading score", "writing score"], axis=1) # independent variable
y = data_encoding[["math score","reading score","writing score"]] # This is a dependent variable
# train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y , train_size=0.8,random_state= 40)
#Model train 

model = LinearRegression()
model.fit(xtrain,ytrain)

predect= model.predict(xtest)
print("Model train succssfully ! simple predection")
print(predect[:5])

pred_data = pd.DataFrame(predect,columns=["math_pre","reading_pre","writing_pre"])
# Acutal ytest and index reste
acutal_pre =ytest.reset_index(drop = True)
finall_comperar= pd.concat([pred_data,acutal_pre],axis=1)
print(finall_comperar.head(10))

# Use the mean square error
mse = mean_squared_error(acutal_pre,predect)
print(f"mean square error:{mse:.2f}")

# predicated Actual visualization
plt.figure(figsize=(10,6))
plt.plot(finall_comperar["math score"],label = "Actual math")
plt.plot(finall_comperar["math_pre"],label = "predicted math",linestyle ='--',color ='red')
plt.title("Math Score: Actual visualization predication")
plt.xlabel("Student Scor")
plt.ylabel("Score")
plt.legend()
plt.show()















