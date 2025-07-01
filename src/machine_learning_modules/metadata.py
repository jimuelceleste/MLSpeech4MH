import pandas as pd 

class Metadata:
	def __init__(self, metadata_file, label_column):
		self.metadata = pd.read_csv(metadata_file)
		self.label_column = label_column
	
	def get_filenames(self):
		filenames = self.metadata['filename']
		return filenames
	
	def get_filenames_by_index(self, index):
		filenames = self.metadata.loc[index, 'filename']
		return filenames

	def get_labels(self):
		labels = self.metadata[self.label_column]
		return labels

	def get_labels_by_index(self, index):
		labels = self.metadata.loc[index, self.label_column]
		return labels

	def get_record_ids(self):
		record_ids = self.metadata['record_id']
		return record_ids

	def get_rows(self, index):
		rows = self.metadata.loc[index]
		return rows

	def get_subject_ids(self):
		subject_ids = self.metadata['subject_id']
		return subject_ids