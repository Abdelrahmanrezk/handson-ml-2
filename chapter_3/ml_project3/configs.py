import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import  BaseEstimator, TransformerMixin

DATA_DIR = "datasets/"

def load_data(file_name, data_dir=DATA_DIR):
	'''
	The function to read CSV file from the main directory
	'''
	df_file = pd.read_csv(data_dir + file_name)
	return df_file


def split_data(features_based, data):
	
	split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
	for train_indeces, test_indeces in split.split(data, data[features_based]):
		train_data = data.iloc[train_indeces]
		test_data = data.iloc[test_indeces]

	return train_data, test_data

def check_split_error(data, train_data, test_data, features_based):
	overall     = np.array(data[features_based].value_counts()        / len(data))
	train = np.array(train_data[features_based].value_counts() / len(train_data))
	test  = np.array(test_data[features_based].value_counts() / len(test_data))

	overall.sort()
	train.sort()
	test.sort()


	result = {'overall': overall,'train': train, 'test': test,
	 'train_error': np.abs(overall - train),
				'test_error': np.abs(overall - test)}
	df_result = pd.DataFrame(result)
	df_result.fillna(0)
	return df_result

class FeatureScale(BaseEstimator, TransformerMixin):
    def __init__(self, num_attr_names, all_attr=True):
        self.all_attr         = all_attr
        self.num_attr_names   = num_attr_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        scaling = MinMaxScaler()
        if self.all_attr:
            X = scaling.fit_transform(X)
        else:
            X, X_split = X[self.num_attr_names[:2]], X[self.num_attr_names[2:]]
            X = scaling.fit_transform(X)
            X = np.c_[X, X_split]
        
        X = pd.DataFrame(X, columns=self.num_attr_names)
        return X

class LeaveOrRemoveOutLiers(BaseEstimator, TransformerMixin):
	def __init__(self, attr, l_r_attr=True):
		self.attr     = attr
		self.l_r_attr = l_r_attr
		
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		if self.l_r_attr:
			return X
		else:
			for col in self.attr:
				X.loc[X.groupby(col)[col].transform('count').lt(11), col] = np.nan
			return X

