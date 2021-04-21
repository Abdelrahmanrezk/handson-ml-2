#!/usr/bin/env python
# coding: utf-8

# # 2-Programming Exercise 2: Logistic Regression Coursera (Classification)
# 
# ## Manual Logistic Regression

# #  Introduction 
# 
# ### The task related to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams
# 
# ### This Task is related to Coursera Machine Learning Course by Andrew NG, but implemnted in Python.
# 
# **Most text used in this notebook from ex2.pdf of Coursera**
# 
# **Look at ex2.pdf to get more intuition about the task**
# 
# **The task will be implemented in three ways and three notebooks and it all about Logistic regression (Classification)**
# 
# - As manual code which pure code.
# - Using Sklearn library
# - Using Tensflow & Keras
# 
# ###   Logistic regression with one variable
# 
# In this exercise,we will implement logistic regression and apply it to two different datasets
# 
# ## Most of code is written to be clean and enhancing with functions

# ## logistic Regression

# ##  Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ##  Handling file
# 
# I would like to change the data from txt to be in csv file.
# 
# **You can find these csv files at csv_files direction.**
# 

# In[2]:


def from_txt_to_csv(file_name, cols_names):
    '''
    Argument:
        file_path that you need to convert to csv file
        cols_names if you would like to saved with columns names
    return:
        True if no error occured of saved operations
    '''
# read the txt file with columns name and save as csv file
    try:
        read_file = pd.read_csv ('csv_files/'+ file_name + '.txt', names=cols_names) 
        read_file.to_csv ('csv_files/' + file_name + '.csv', index=None)
# if there is any error print it in separate file in dir logs_files
    except Exception as e:
        file = open("log_files/from_txt_to_csv.log","+a")
        file.write("This error related to function from_txt_to_csv function of manual_linea_regression notebook\n" 
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
    return True


# In[3]:


# create columns name for our data
cols = ['exam_1', 'exam_2', 'admitted']
# call the function
from_txt_to_csv('ex2data1', cols)


# ##  Read the csv file

# In[4]:


df_file = pd.read_csv('csv_files/ex2data1.csv')
df_file.head()


# ## Visualizing the data
# 
# - first retrive the admitted students as positive_points
# - second retrive the Non-admitted students negative_points
# - Visualizing the result

# In[5]:


positive_points = df_file[df_file['admitted'] == 1]
print("Some of positive points\n", positive_points[:5])


# In[6]:


negative_points = df_file[df_file['admitted'] == 0]
print("Some of positive points\n", negative_points[:5])


# In[7]:


def init_2d_graphs(*colors):
    '''
        Just graph initialize in good way
    '''
    plt.style.use(colors) # color of your 2d graph
    plt.figure(figsize=(10,6)) # set the figure size
    return True


# In[8]:


init_2d_graphs('ggplot', 'dark_background' )

plt.scatter(positive_points['exam_1'],positive_points['exam_2'], s = 200, c = 'green', marker = 'P', label =  'Admitted')
plt.scatter(negative_points['exam_1'],negative_points['exam_2'], s = 200, c = 'red', marker = 'x', label =  'Not-Admitted')
plt.title("The Relation between " + 'Admitted Student' + " And " + 'Not-Admitted Student' )
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')


# ## Prepare The Sigmoid function
# 
# **Before you start with the actual cost function, recall that the logistic regres-sion hypothesis is defined as:**
# 
# ![alt text](images/h_x.png "Hypothesis")
# 
# **where functiongis the sigmoid function.  The sigmoid function is defined as:**
# 
# ![alt text](images/sigmoid_1.png "sigmoid_1")

# In[9]:


def sigmoid(z):
    '''
    Argument:
        z = x*theta
    '''
# np.log is natural log of base 2 which is called ln
    g_z = 1/(1+np.exp(-z))
    return g_z


# ## Visualizing the sigmoid

# In[10]:


x = np.arange(-10, 11) #from -10 to 10, 11 because of 0 index
g_z = sigmoid(x)


# In[11]:


init_2d_graphs('ggplot', 'dark_background' )
plt.scatter(x,g_z, s=200, c='red', marker='o')
plt.plot(x, g_z, 'red', label= 'Sigmoid Function',linewidth=3)
plt.xlabel('The range of values')
plt.ylabel('The sigmoid values')
plt.title('Visualizing of sigmoid function')
plt.legend()


# ## parameters initialize

# In[12]:


x = np.array(df_file.iloc[:, :-1])
y = np.array(df_file.iloc[:, -1])
m = len(y)
y = y.reshape(-1,1)
X = np.c_[x, np.ones(m)] # add x0 = 1

all_costs = []
all_thetas = []

thetas = np.zeros((3,1)) # initialize threats as 2d array and 2*1 dimension with 0 values
print('Theats shape is: ', thetas.shape)
print("#" * 70)
print('Theats values are: ', thetas)
print("#" * 70)

print('First 5 values now of the features are: ', X[:5])


# ## features normalization or Data Scaling

# In[13]:


def features_normalization_with_std(X):
    '''
        Normalize the data via standard deviation
    '''
    X= (X - np.mean(X)) / np.std(X)
    return X


# In[14]:


X = features_normalization_with_std(X)


# ## Cost function
# 
# **Recall that the cost function in logistic regression is:**
# 
# ![alt text](images/cost_func1.png "cost_function")

# In[15]:


def cost_function(thetas,x,m,y):
    '''
    Arguments:
        thetas the paramter we need to minimize of shape 3*1
        x the eatures of our dataset 100*3
        m number of training examples 
        y is output we need to predict
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''
    z   = np.matmul(x,thetas) # hypothesis function
    y_hat = sigmoid(z)
    
# get the cost function
    cost_function = (-1/m) * (np.matmul(y.T, np.log(y_hat)) + np.matmul((1-y).T, np.log(1-y_hat)))
    return cost_function


# In[16]:


J = cost_function(thetas, X, m, y)
all_costs.append(J)
print("The cost funtion of our training data is: ", J)


# ## Gradient Descent
# 
# **Recall that the Gradient Descent vectorized implementation of classification is :**
# 
# ![alt text](images/vecto_grad.png "vecto_grad")
# 

# In[17]:


def gradient_descent(thetas,x,m,y, learning_rate):
    '''
    Arguments:
       thetas the paramter we need to minimize of shape 3*1
        x the eatures of our dataset 100*3
        m number of training examples 
        y is output we need to predict
        learning rate is alpha which inilized above as .01
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''
    z = np.matmul(x,thetas)
    y_hat = sigmoid(z)
    
    grad = (learning_rate / m) * (np.matmul(x.T, y_hat - y))
    
# return the gradient but transposed to be 2*1 instead of 1*2 to that maps to theta dimensions
    return grad


# In[18]:


grad = gradient_descent(thetas, X, m, y, .01)
all_thetas.append(grad)
print("Instead of Thetas as zero now thetas paramters after just 1 iteration is: ", grad.shape)


# In[19]:


def logistic_regression_model(thetas, x, m, y, learning_rate, num_of_iterations):
    '''
    Arguments:
        thetas the paramter we need to minimize of shape 2*1
        x the eatures of our dataset 97*2
        m number of training examples
        y is output we need to predict
        learning rate is alpha which inilized above as .01
        num_of_iterations you need to minimize the cost function
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''
    for i in range(num_of_iterations):
        J = cost_function(thetas, x, m, y)
        all_thetas.append(thetas)
        all_costs.append(J)
# get new values of theta as gradient descent step
        grad = gradient_descent(thetas, x, m , y, learning_rate)

# update theta so if grad is negative the theta will increase otherwise will decrease
        thetas = thetas - grad

    return all_costs, thetas, all_thetas


# In[20]:


all_costs, last_thetas, all_theta = logistic_regression_model(thetas, X, m, y, .01, 500)


# In[21]:


J = cost_function(last_thetas, X, m, y)
print("Our cost function after 500 iterations is: ", J)


# In[22]:


cost_list = np.array(all_costs)
cost_list = cost_list.reshape(-1,1)


# In[23]:


init_2d_graphs('ggplot', 'dark_background' )
plt.plot(np.arange(501), cost_list, 'red', label= 'Cost Function',linewidth=3)
plt.xlabel('The number of iterations')
plt.ylabel('The cost of after 500 iteration')
plt.title('Error vs. Training Iterations')
plt.legend()


# In[24]:


all_costs, last_thetas, all_theta = logistic_regression_model(thetas, X, m, y, .1, 500)


# In[25]:


cost_list = np.array(all_costs)
cost_list = cost_list.reshape(-1,1)


# In[26]:


init_2d_graphs('ggplot', 'dark_background' )
plt.plot(np.arange(len(cost_list)), cost_list, 'red', label= 'Cost Function',linewidth=3)
plt.xlabel('The number of iterations')
plt.ylabel('The cost of after another 500 iteration with learning rate of .1')
plt.title('Error vs. Training Iterations')
plt.legend()


# ## Regularized logistic regression
# 
# In this part of the exercise, we will implement regularized logistic regressionto predict whether microchips from a fabrication plant passes quality assur-ance (QA). 

# ## Handling file
# 
# **As step above**

# In[27]:


# create columns name for our data
cols = ['test_1', 'test_2', 'Accepted']
# call the function
from_txt_to_csv('ex2data2', cols)
df_file = pd.read_csv('csv_files/ex2data2.csv')
#now you can see the data after convert to csv with columns name
df_file.head()


# In[28]:


df_file.describe()


# ## Visualizing the data
# 

# In[29]:


positive_points = df_file[df_file['Accepted'] == 1]
print("Some of positive points\n", positive_points[:5])


# In[30]:


negative_points = df_file[df_file['Accepted'] == 0]
print("Some of positive points\n", negative_points[:5])


# In[31]:


init_2d_graphs('ggplot', 'dark_background' )

plt.scatter(positive_points['test_1'],positive_points['test_2'], s = 200, c = 'green', marker = 'P', label =  'Accepted')
plt.scatter(negative_points['test_1'],negative_points['test_2'], s = 200, c = 'red', marker = 'x', label =  'Not-Accepted')
plt.title("The Relation between " + 'Accepted Test' + " And " + 'Not-Accepted Student' )
plt.legend()
plt.xlabel('Test 1 ')
plt.ylabel('Test 2 ')


# ## Feature mapping
# 
# **Some times you increase your features to fit the data well**

# In[32]:


x = np.array(df_file.iloc[:, :-1])
y = np.array(df_file.iloc[:, -1])
m = len(y)
y = y.reshape(-1,1)

all_costs = []
all_thetas = []


# In[33]:


for i in range(1,6):
    for j in range(i):
        '''
        mapping features is to make polynomial features from linear here is to power 4
        '''
        df_file['F' + str(i) + str(j)] = np.power(x[:, 0], i-j) * np.power(x[:, 1], j)


# In[34]:


df_file.head()


# In[35]:


x = np.array(df_file.iloc[:, 3:])
X = np.c_[x, np.ones(m)] # add x0 = 1
thetas = np.zeros((X.shape[1],1)) # initialize threats as 2d with number of features


# ## Regularized Cost function
# 
# **we will use the cost function created above and add the Regularized part instead of write the function again**
# 
# **Recall that the Regularized cost function in logistic regression is:**
# 
# ![alt text](images/reg_cost.png "Regularized cost")

# In[36]:


J = cost_function(thetas, X, m, y)
J += (1/(2*m)) * np.sum(np.power(thetas[1:] , 2))
all_costs.append(J)
print("The cost funtion of our training data is: ", J)


# ## Regularized Gradient Descent
# 
# **we will use the Gradient Descent function created above and add the Regularized part instead of write the function again**
# 
# **Recall that the Regularized cost function in logistic regression is:**
# 
# ![alt text](images/grad_regulized.png "Regularized Gradient")

# In[37]:


grad = gradient_descent(thetas, X, m, y, .01)
grad += (1/m) * thetas
all_thetas.append(grad)
print("Instead of Thetas as zero now thetas paramters after just 1 iteration is: ", grad)


# In[38]:


def logistic_regression_model_regularized(thetas, x, m, y, learning_rate, num_of_iterations):
    '''
    Arguments:
        thetas the paramter we need to minimize of shape 2*1
        x the eatures of our dataset 97*2
        m number of training examples
        y is output we need to predict
        learning rate is alpha which inilized above as .01
        num_of_iterations you need to minimize the cost function
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''
    for i in range(num_of_iterations):
        J = cost_function(thetas, x, m, y)
        J += (1/(2*m)) * np.sum(np.power(thetas[1:] , 2))
        
        all_thetas.append(thetas)
        
        all_costs.append(J)
# get new values of theta as gradient descent step
        grad = gradient_descent(thetas, x, m , y, learning_rate)
        grad += (1/m) * thetas
# update theta so if grad is negative the theta will increase otherwise will decrease
        thetas = thetas - grad

    return all_costs, thetas, all_thetas


# In[39]:


all_costs, last_thetas, all_theta = logistic_regression_model_regularized(thetas, X, m, y, .01, 500)


# In[40]:


cost_list = np.array(all_costs)
cost_list = cost_list.reshape(-1,1)
cost_list.shape


# In[41]:


init_2d_graphs('ggplot', 'dark_background' )
plt.plot(np.arange(501), cost_list, 'red', label= 'Cost Function',linewidth=3)
plt.xlabel('The number of iterations')
plt.ylabel('The cost of after 500 iteration')
plt.title('Error vs. Training Iterations')
plt.legend()


# In[42]:


all_costs, last_thetas, all_theta = logistic_regression_model(thetas, X, m, y, 1, 10000)


# In[43]:


cost_list = np.array(all_costs)
cost_list = cost_list.reshape(-1,1)
cost_list.shape


# In[45]:


init_2d_graphs('ggplot', 'dark_background' )
plt.plot(np.arange(10501), cost_list, 'red', label= 'Cost Function',linewidth=3)
plt.xlabel('The number of iterations')
plt.ylabel('The cost of after 10000 iteration')
plt.title('Error vs. Training Iterations')
plt.legend()


# ## The last cost function

# In[46]:


J = cost_function(last_thetas, X, m, y)
J += (1/(2*m)) * np.sum(np.power(thetas[1:] , 2))
print("The cost funtion of our training data is: ", J)


# In[ ]:




