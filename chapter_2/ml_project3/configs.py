from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os




LOCAL_DATA_DIR = "datasets/"


class AddCombinedAttributes(BaseEstimator, TransformerMixin):
	"""docstring for CombinedAttributes"""
	def __init__(self, add_GDP_for_family = True):
		self.add_GDP_for_family = add_GDP_for_family
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		if self.add_GDP_for_family:
			GDP_for_family = X[:, 2] / (X[:, 3]+1)
			X = np.c_[X, GDP_for_family]
		return X

def load_csv_file(file_name, load_data_dir=LOCAL_DATA_DIR):
	"""
	The function to read csv file.
	Argument:
	load_data_dir: Take default value for the direction where the data is.
	file_name: Which file from this direction you need to read.
	"""

	df_file = pd.read_csv(load_data_dir + file_name)
	return df_file


def stratified_splitting(data, split_attr):
	"""
	The function to split data based on specific attribute 
	to ensure that we include all the categories of the data in our test set.
	"""
	split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
	for tr_index, tes_index in split.split(data, data[split_attr]):
		tr_stratified_set = data.loc[tr_index]
		tes_stratified_set = data.loc[tes_index]

	return tr_stratified_set, tes_stratified_set 
	
def compare_random_stratified_split(data, random_set, stratified_set, split_attr):
	overall          = data[split_attr].value_counts() / len(data)
	random_split     = random_set[split_attr].value_counts() / len(random_set)
	stratified_split = stratified_set[split_attr].value_counts() / len(stratified_set)

	# Sort values
	overall.sort_values(ascending=False)
	random_split.sort_values(ascending=False)
	stratified_split.sort_values(ascending=False)

	overall_Vs_random_error     = np.abs(overall - random_split)
	overall_Vs_stratified_error = np.abs(overall - stratified_split)

	# As it all numpy array I will transform to pandas dataframe
	dict_result = {"overall": overall, "random_split": random_split, "stratified_split": stratified_split,
              "overall_Vs_random_error": overall_Vs_random_error,
               "overall_Vs_stratified_error": overall_Vs_stratified_error}
	error_result = pd.DataFrame(dict_result, columns=dict_result.keys())
	return error_result


def visualize_plot(data,plot_kind, x_axis, y_axis, point_size=50, color="gray", colorbar=False):
	plt.figure(figsize=(10,10))
	plt.style.use('fivethirtyeight')
	data.plot(kind=plot_kind, x=x_axis, y=y_axis, s=point_size, 
		c=color, cmap=plt.get_cmap("jet"), colorbar=colorbar)

	return True


def split_seprate(data):
	happiness_report_num = data.drop(['Happiness Score', 'Region', 'Country'], axis=1)
	happiness_report_cat = data[['Region']]
	happiness_report_labels = data[['Happiness Score']]

	return happiness_report_num, happiness_report_cat, happiness_report_labels


def prepare_data(data, target='Happiness Score', number_of_instances=10):

	some_data = data.iloc[:number_of_instances].copy()
	some_data_labeled = some_data[[target]]
	some_data = some_data.drop(target, axis=1)

	return some_data,  some_data_labeled