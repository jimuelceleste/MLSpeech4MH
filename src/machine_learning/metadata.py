import os 

import pandas as pd 

class Metadata:
	"""
	Class that represents the metadata file.

	Parameters
	----------
	metadata_file : str 
		path to the metadata file
	metadata_config : dict
		metadata configuration

	Attributes
	----------
	metadata_file : str
		path to the metadata file
	metadata : Pandas DataFrame
		Pandas DataFrame instance of the metadata csv file
	label_column : str
		column name of the label being predicted
	record_id_column : str 
		column name of the record id 
	subject_id_column : str 
		column name of the subject_id 
	filename_column : str 
		column name of the feature filenames

	Methods 
	-------
	__init__(input_dir, metadata_config)
		Constructor
	get_filenames()
		Returns the filenames column.
	get_filenames_by_index(index)
		Returns the filenames column of specified index.
	get_labels()
		Returns the labels column.
	get_record_ids()
		Returns the record_id column.
	get_rows_by_index(index)
		Returns rows with specified index.
	get_subject_ids()
		Returns the subject_id column.
	get_features(features_dir)
		Returns features Pandas DataFrame.
	"""

	def __init__(self, input_dir, metadata_config):
		"""Constructor
		
		Parameters
		----------
		input_dir : str 
			path to the input directory
		metadata_config : dict 
			metadata configuration
		"""

		self.metadata_file = os.path.join(
			input_dir,
			metadata_config['file']
		)
		self.metadata = pd.read_csv(self.metadata_file)
		self.label_col = metadata_config['label_col']
		self.record_id_col = metadata_config['record_id_col']
		self.subject_id_col = metadata_config['subject_id_col']
		self.filename_col = metadata_config['filename_col']
	
	def get_filenames(self):
		"""Returns the filenames column."""
		return self.metadata[self.filename_col]
	
	def get_filenames_by_index(self, index):
		"""Returns the filenames column of specified index.
		
		Parameters
		----------
		index : list 
			list of indeces to retrieve
		"""
		filenames = self.metadata.loc[index, self.filename_col]
		return filenames

	def get_labels(self):
		"""Returns the labels column."""
		labels = self.metadata[self.label_col]
		return labels

	def get_labels_by_index(self, index):
		"""Returns the labels column of specified index.

		Parameters
		----------
		index : list 
			list of indeces to retrieve
		"""
		labels = self.metadata.loc[index, self.label_col]
		return labels

	def get_record_ids(self):
		"""Returns the record_id column."""
		record_ids = self.metadata[self.record_id_col]
		return record_ids

	def get_rows_by_index(self, index):
		"""Returns rows with specified index.
		
		Parameters
		----------
		index : list 
			list of indeces to retrieve
		"""
		rows = self.metadata.loc[index]
		return rows

	def get_subject_ids(self):
		"""Returns the subject_id column."""
		subject_ids = self.metadata[self.subject_id_col]
		return subject_ids

	def get_features(self, features_dir):
		"""Returns features Pandas DataFrame.
		
		Parameters
		----------
		features_dir : str
			path to the features directory
		
		Returns 
		-------
		object
			Pandas DataFrame of all the features.
		"""
		filenames = self.get_filenames()
		
		features_list = []
		for file in filenames:
			features_file = os.path.join(features_dir, file)
			features = pd.read_csv(features_file)
			features_list.append(features)	
		features_df = pd.concat(features_list)
		
		return features_df