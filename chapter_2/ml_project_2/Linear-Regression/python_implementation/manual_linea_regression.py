#!/usr/bin/env python
# coding: utf-8

# # Programming Exercise 1: Linear Regression Coursera

# # 1.1 Introduction 
# 
# ### This Task is related to Coursera Machine Learning Course by Andrew NG, but implemnted in Python.
# 
# **Most text used in this notebook from ex1.pdf of Coursera**
# 
# **power point slides beside of code that depends on Material of Coursera but more enhancing:**
# 
# - What is ML
# - Supervised Learning
# - Linear Regresison
# - Fitting Line
# - Regression Apps
# - Equation of Regression & Slope behind of Linear Equation
# - Different types of slope
# - Linear Regression Notations
# - simple graphs with differnt notations
# - cost function
# - parameters and hyperparameters
# - Gradient Descent
# - Linear Algebra
# - Univariate & Multi features
# - Vectorization instead of loops
# - Feature Scaling & Normalization
# 
# **snapshot from our slides**
# 
# ![alt text](images/differnt_notations.png "differnt_notations")
# 
# 
# **The task will be implemented in three ways and three notebooks and it all about linear regression**
# 
# - As manual code which pure code.
# - Using Sklearn library
# - Using Tensflow & Keras
# 

# ## 1.2 Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### 1. 3 linear regression with one variable
# 
# In this part of this exercise,  we will implement linear regression with one variable to predict profits for a food truck.
# 
# Suppose you are the CEO of arestaurant  franchise  and  are  considering  different  cities  for  opening  a  newoutlet.  The chain already has trucks in various cities and you have data forprofits and populations from the cities.
# 
# You would like to use this data to help you select which city to expandto next.
# 
# The fileex1data1.txt contains the dataset for our linear regression prob-lem.  The first column is the population of a city and the second column is the profit of a food truck in that city.  A negative value for profit indicates a loss.

# ## 1.4 Handling file
# 
# I would like to change the data from txt to be in csv file, and at the end I will provide you with these csv file to ignore all of the above code and start with the csv file.
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
# read the txt file with columns name
    try:
        read_file = pd.read_csv ('ex1/'+ file_name + '.txt', names=cols_names) 
    
    # save as csv to our folder csv_files
        read_file.to_csv ('csv_files/' + file_name + '.csv', index=None)
    except Exception as e:
        file = open("log_files/from_txt_to_csv.log","+a")
        file.write("This error related to function from_txt_to_csv function of manual_linea_regression notebook\n" 
                   + str(e) + "\n" + "#" *99 + "\n") # "#" *99 as separated lines
    return True


# In[3]:


# create columns name for our data
cols = ['city_population', 'food_truck_profit']
# call the function
from_txt_to_csv('ex1data1', cols)


# In[4]:


df_file = pd.read_csv('csv_files/ex1data1.csv')
# now you can see the data after convert to csv with columns name
df_file.head()


# ### 1.5  Plotting the Data
# Before  starting  on  any  task,  it  is  often  useful  to  understand  the  data  by visualizing it.  For this dataset,  you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). (Many other problems that you will encounter in real life are multi-dimensional and can’t be plotted on a 2-d plot.)

# ### Graph initialize

# In[5]:


def init_2d_graphs(*colors):
    '''
        Just graph initialize in good way
    '''
    plt.style.use(colors)
    plt.figure(figsize=(10,6))
    return True


# In[6]:


def ploting_2d_data(x_axis, y_axis, *arg):
    '''
    Argument:
        x_axis, y_axis of the graph
        argv as typle of values:
            arg[0] = xlabel
            arg[1] = ylabel
            arg[2] = point_size as you see different size of point because of using random * with value
            arg[3] = point_color as you see red points
            arg[4] = marker_type as you see +
            arg[5] = legend name as you see Training data
    '''
# the labels of x & y
    plt.xlabel(arg[0])
    plt.ylabel(arg[1])
# argv[3] * np.random.rand(5) different point size
# 
    plt.scatter(x,y, s = arg[2], c = arg[3], marker = arg[4], label = arg[5])
    plt.title("The Relation between " + str(arg[0]) + " And " + str(arg[1]))
    plt.legend()
    return "2d Graph Scatter representation"


# In[7]:


# our initialized and graph draw for our dataset
x = df_file['city_population'] # x_axis
y = df_file['food_truck_profit'] # y_axis
# because y is shape (97,) whcih rank of 0 and we need to be (97,1) to subtract from y_hat
y = y.values.reshape(len(x),1) # because of type Series we use .values

x_label = 'Population of City in 10,000s'
y_label = 'Profit in $10,000s'
graph_legend = 'Training data'

init_2d_graphs('ggplot', 'dark_background' )
ploting_2d_data(x,y, x_label, y_label, 300, 'red', 'P', graph_legend)


# ### some static of our data set we can see

# In[8]:


df_file.describe()


# ### 1.6  Gradient Descent
# In  this  part,  we  will  fit  the  linear  regression  parameters θ to  our  datasetusing gradient descent.
# The objective of linear regression is to minimize the cost function

# ## Implementation Steps
# 
# first we have x and we need to map it to y
# - x = Population of City in 10,000s
# - y = Profit in $10,000s
# 
# **Hypothesis**
# 
# - second the hypothesis function is y_hat = theta0 + theta1 * X1 and we initialize X0 = 1 which have no effect on theta0:
# 
# ![alt text](images/hypothesis_linear.png "hypothesis_linear_function")
# 
# - so we need to inilize these theats.
# 
# **Cost function**
# 
# 
# - Third the cost function is use m as training example so we need to get the number of our training examples:
# ![alt text](images/cost_function.png "cost_function")
# 
# **Gradient Steps**
# 
# - Fourth in gradient descent there is another parameters alpha we need to initialize which the learning rate.
# 
# - Fifth and the last we need to specify the number of iteration will used to iterate and update the parameters:
# 
# ![alt text](images/paramters_updated.png "paramters_updated")
# 

# In[9]:


# variables and parameters initialize
m = len(x) 
print("Number of training example: ", m)
Alpha = .01 # learning rate
iterations = 1500 # number of gradient descent iterations
thetas = np.zeros((2,1)) # initialize threats as 2d array and 2*1 dimension with 0 values
print('Theats shape is: ', thetas.shape)
print('Theats values are: ', thetas)
# as we said above we need to expand x to have x0=1 for each training axample  because of theta 0.
X = np.stack((x, np.ones(m)), axis=1) # create x0 = 1 for each example
print("Now X shape is: ", X.shape)
print("Now first 5 element of X is", X[:5,:]) # all columns in first 5 rows


# ## 1.7 Cost Function
# 
# As you perform gradient descent to learn minimize the cost function J(θ),it  is  helpful  to  monitor  the  convergence by computing the cost. In  thissection, you will implement a function to calculate J(θ) so you can check the convergence of your gradient descent implementation.
# 
# ### Remember the cost function is:
# 
# ![alt text](images/cost_function.png "cost_function")
# 
# **theta is 2 * 1 and X is 97 * 2 so we can multiply X*theta and get 97 * 1**

# In[10]:


def cost_function(thetas,x,m,y):
    '''
    Arguments:
        thetas the paramter we need to minimize of shape 2*1
        x the eatures of our dataset 97*2
        m number of training examples
        y is output we need to predict
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''


# get h_x first or called y_hat
# y_hat = theta0 * x0 + theta1 * x1 and with vectorized will be x = 97*2 * theta =  2*1
    y_hat = np.matmul(x,thetas)
# get the cost function
    cost_function = (1/(2*m)) * np.sum(np.square(y_hat - y))
    return cost_function


# In[11]:


J = cost_function(thetas, X, m, y)
print("The cost funtion of our training data is: ", J)


# ## 1.8 Gradient descent
# 
# Next, we will implement gradient descent.
# 
# ### Remember the  Gradient descent is:
# 
# ![alt text](images/paramters_updated.png "paramters_updated")
# 

# In[12]:


def gradient_descent(thetas,x,m,y, learning_rate):
    '''
    Arguments:
        thetas the paramter we need to minimize of shape 2*1
        x the eatures of our dataset 97*2
        m number of training examples
        y is output we need to predict
        learning rate is alpha which inilized above as .01
    return:
        cost function as total squared cost of our predicted values h_x and the real values y
    '''

# get h_x first or called y_hat
# y_hat = theta0 * x0 + theta1 * x1 and with vectorized will be x = 97*2 * theta =  2*1
    y_hat = np.matmul(x,thetas)

# get the gradient descent thetas
# Transpose and multiply via vectorize so no need to summation because:
# y_hat = 97*1 - y = 97*1 will be 97*1 and multiply by x which 97*2 so need to transpose to be 1*97 and x 97*2
    cost = np.matmul(np.transpose(y_hat-y), x) 
    grad =((learning_rate/m) * cost)
    
# return the gradient but transposed to be 2*1 instead of 1*2 to equal theta dimensions
    return grad.T


# In[13]:


grad = gradient_descent(thetas, X, m, y, Alpha)
print("instead of Thetas as zero now thetas paramters after just 1 iteration is: ", grad)


# In[14]:


def linear_regression_model(thetas, x, m, y, learning_rate, num_of_iterations):
    costs = []
    all_theta = []
    for i in range(num_of_iterations):
        J = cost_function(thetas, x, m, y)
        all_theta.append(thetas)
        costs.append(J)
# get new values of theta as gradient descent step
        grad = gradient_descent(thetas, x, m , y, Alpha)

# update theta so if grad is negative the theta will increase otherwise will decrease
        thetas = thetas - grad

    return costs, thetas, all_theta


# In[15]:


all_cost, last_thetas, all_theta = linear_regression_model(thetas, X, m, y, Alpha, iterations)


# In[16]:


J = cost_function(last_thetas, X, m, y)
print("Our cost function after 1500 iterations is: ", J)


# In[17]:


predict1 = np.abs(np.matmul([1, 3.5],last_thetas))
predict2 = np.abs(np.matmul([1, 7],last_thetas))
print("Our Prediction 1", predict1)
print("Second Prediction", predict2)


# ## 1.9 Graphs of fitting line
# 
# After we has train our model and update the paramters with new values we need to fitting the line to our data.
# 
# - So first we get the predicted value y hat with new update thetas as final result
# - second plot multiple fitting lines to show differnt values of thetas
# - Third print the x values and real y to see the difference between real points and fitting line

# In[18]:


# Plot the graph with different first 4 values of thetas and last values of thetas
init_2d_graphs('ggplot', 'dark_background' ) # initialize graphics size
for i in range(4):
    y_hat = np.matmul(X, all_theta[i])
    plt.plot(x, y_hat, label='predict ' + str(i+1), linewidth=3)
y_hat = np.matmul(X, last_thetas)
plt.plot(x, y_hat, label= 'last predict',linewidth=2)
ploting_2d_data(x,y, x_label, y_label, 200, 'yellow', 'X', graph_legend)


# ## 2.0 Cost function graph
# 
# after we see how the fitting line on our data its usful to see how cost function decreased with differnt step of gradient descent.

# In[19]:


init_2d_graphs() # initialize graphics size
#np.arange(iters) means from 0 to 1500 iterations

plt.plot(np.arange(iterations), all_cost, 'r', label='Cost Functions', linewidth=2) 
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title('Error vs. Training Iterations')
plt.legend()


# ### Note
# **as you can see in graph above its will be fine to stop after 400 or 600 iterations because it has a small decrease**

# # 2.1 linear regression with multiple variables
# 
# In this part, we will implement linear regression with multiple variables to predict  the  prices  of  houses.
# 
# Suppose you are selling your house and you want to know what a good market price would be.  One way to do this is to first collect information on recent houses sold and make a model of housing prices.

# ## 2.2 Handling file
# 
# **As above step**
# I would like to change the data from txt to be in csv file, and at the end I will provide you with these csv file to ignore all of the above code and start with the csv file.
# 

# In[20]:


# create columns name for our data
cols = ['house_size', 'number_of_bedrooms', 'house_price']
# call the function
from_txt_to_csv('ex1data2', cols)
df_file = pd.read_csv('csv_files/ex1data2.csv')
#now you can see the data after convert to csv with columns name
df_file.head()


# In[21]:


df_file.describe()


# ## 2.3 features normalization or Data Scaling
# 
# **Its important step to make the values of different features within spceific range because it help you in:**
# 
# - Avoid NAN values because numbers in operations of multiplication
# - its help the machine to deal with numbers within range than different ranges and the operations be less cost
# 
# **try to comment the line of calling function features_normalization_with_std and see result of how its affect.**
# 
# ![alt text](images/nan.png "Nan Value")
# 
# 
# **Two function implemented for feature scaling choose any of them**

# In[22]:


def features_normalization_with_std(X):
    '''
        Normalize the data via standard deviation
    '''
    X= (X - np.mean(X)) / np.std(X)
    return X


# In[23]:


def features_normalization_with_min_max(X):
    '''
        Normalize the data via min max approach
    '''
    X = (X - np.mean(X)) / (np.max(X) - np.min(X))
    return X


# In[24]:


df_file = features_normalization_with_std(df_file)


# In[25]:


x = np.array(df_file.iloc[:, :2])# get the 2 features columns)
y = df_file['house_price'] # the real output 

# # because y is shape (97,) whcih rank of 0 and we need to be (97,1) to subtract from y_hat
y = y.values.reshape(len(y),1) # because of type Series we use .values

# variables and parameters initialize
all_cost = []
m = len(y) 
print(x.shape)
print("#"*80)
print("Number of training example: ", m)
print("#"*80)
Alpha = .1 # learning rate
iterations = 100 # number of gradient descent iterations
thetas = np.zeros((3,1)) # initialize threats as 2d array and 2*1 dimension with 0 values
print('Theats shape is: ', thetas.shape)
print("#"*80)
print('Theats values are: ', thetas)
print("#"*80)
X = np.column_stack((x,np.ones(len(y))))
print("Now X shape is: ", X.shape)


# In[26]:


print("Now the first 5 rows of x values are: ", X[:5, :])


# ## 2.4 call the Cost Function

# In[27]:


J = cost_function(thetas, X, m, y)
print("The cost funtion after data scaling is: ", J)


# ## 2.4 call the gradient descent

# In[28]:


grad = gradient_descent(thetas, X, m, y, Alpha)
print("instead of Thetas as zero now thetas paramters after just 1 iteration is: ", grad)


# ## 2.4 call the linear_regression_model for multiple of iterations

# In[29]:


all_cost, last_thetas, all_theta = linear_regression_model(grad, X, m, y, Alpha, iterations)


# In[30]:


J = cost_function(last_thetas, X, m, y)
print("Our cost function after 1000 iterations without feature scaling: ", J)


# ## 2.5 Cost function graph

# In[31]:


init_2d_graphs() # initialize graphics size
#np.arange(iters) means from 0 to 1500 iterations
plt.plot(np.arange(iterations), all_cost, 'r', label='Cost Functions', linewidth=2) 
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title('Error vs. Training Iterations')
plt.legend()


# ## Another with features_normalization_with_min_max

# In[32]:


# create columns name for our data
cols = ['house_size', 'number_of_bedrooms', 'house_price']
# call the function
from_txt_to_csv('ex1data2', cols)
df_file = pd.read_csv('csv_files/ex1data2.csv')
#now you can see the data after convert to csv with columns name
df_file.head()


# In[33]:


df_file = features_normalization_with_min_max(df_file)


# In[34]:


x = np.array(df_file.iloc[:, :2])# get the 2 features columns)
y = df_file['house_price'] # the real output 

# # because y is shape (97,) whcih rank of 0 and we need to be (97,1) to subtract from y_hat
y = y.values.reshape(len(y),1) # because of type Series we use .values

# variables and parameters initialize
all_cost = []
m = len(y) 
print(x.shape)
print("#"*80)
print("Number of training example: ", m)
print("#"*80)
Alpha = .1 # learning rate
iterations = 100 # number of gradient descent iterations
thetas = np.zeros((3,1)) # initialize threats as 2d array and 2*1 dimension with 0 values
print('Theats shape is: ', thetas.shape)
print("#"*80)
print('Theats values are: ', thetas)
print("#"*80)
X = np.column_stack((x,np.ones(len(y))))
print("Now X shape is: ", X.shape)


# In[35]:


J = cost_function(thetas, X, m, y)
print("The cost funtion after data scaling is: ", J)


# In[36]:


grad = gradient_descent(thetas, X, m, y, Alpha)
print("instead of Thetas as zero now thetas paramters after just 1 iteration is: ", grad)


# In[37]:


all_cost, last_thetas, all_theta = linear_regression_model(grad, X, m, y, Alpha, iterations)


# In[38]:


J = cost_function(last_thetas, X, m, y)
print("Our cost function after 1000 iterations without feature scaling: ", J)


# In[39]:


init_2d_graphs() # initialize graphics size
#np.arange(iters) means from 0 to 1500 iterations
plt.plot(np.arange(iterations), all_cost, 'r', label='Cost Functions', linewidth=2) 
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title('Error vs. Training Iterations')
plt.legend()


# In[ ]:





# In[ ]:




