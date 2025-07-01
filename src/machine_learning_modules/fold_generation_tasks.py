import os 
import pickle 
import shutil

import luigi 
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from typing_extensions import override

from machine_learning_modules.metadata import Metadata
from utility_modules.file import File

class FoldGenerationTask(luigi.Task):
	"""
	Generates outer and inner folds for nested cross-validation.

	Parameters:
		input_dir (str): Path to the input directory.
		output_dir (str): Path to output directory.
		n_splits (int): Number of folds to generate.
		random_state (int): Random seed for the fold generation.
		shuffle (bool): True if the samples will be shuffled. 
		metadata_label_column (str): Label column inside the metadata file.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	n_splits = luigi.IntParameter()
	random_state = luigi.IntParameter()
	shuffle = luigi.BoolParameter()
	metadata_label_column = luigi.Parameter()
	metadata_filename = luigi.Parameter()

	@override
	def requires(self):
		metadata_file = os.path.join(self.input_dir, self.metadata_filename)
		yield File(metadata_file)

	@override
	def run(self):
		splitter = self.get_splitter()
		metadata = self.get_metadata()
		folds = self.generate_folds(splitter, metadata)
		self.save_folds(folds, metadata)
		self.save_iterator(folds)

	@override
	def output(self):
		output_files = []

		# for all folds: train.csv and validate.csv
		for fold in range(self.n_splits):
			train_file = os.path.join(self.output_dir, f"fold_{fold}", "train.csv")
			validate_file = os.path.join(self.output_dir, f"fold_{fold}", "validate.csv")
			
			output_files.append(luigi.LocalTarget(train_file))
			output_files.append(luigi.LocalTarget(validate_file))

		# iterator.pkl
		iterator_file = os.path.join(self.output_dir, "iterator.pkl")
		output_files.append(luigi.LocalTarget(iterator_file))

		return output_files
	
	def get_splitter(self):
		pass

	def get_metadata(self):
		metadata_file = os.path.join(self.input_dir, self.metadata_filename)
		metadata = Metadata(metadata_file, self.metadata_label_column)
		return metadata

	def generate_folds(self, splitter, metadata):
		record_ids = metadata.get_record_ids()
		subject_ids = metadata.get_subject_ids()
		labels = metadata.get_labels()
		
		folds = splitter.split(
			X=record_ids, 
			y=labels, 
			groups=subject_ids
		)

		folds_list = [(train, validate) for train, validate in folds]
		
		return folds_list

	def save_folds(self, folds, metadata):
		for fold, (train, validate) in enumerate(folds):
			current_output_dir = os.path.join(self.output_dir, f"fold_{fold}")
			
			train_file = os.path.join(current_output_dir, "train.csv")
			validate_file = os.path.join(current_output_dir, "validate.csv")
				
			train_df = metadata.get_rows(train)
			validate_df = metadata.get_rows(validate)

			os.makedirs(current_output_dir, exist_ok=True)
			train_df.to_csv(train_file, index=False)
			validate_df.to_csv(validate_file, index=False)

	def save_iterator(self, iterator):
		iterator_file = os.path.join(self.output_dir, "iterator.pkl")
		with open(iterator_file, 'wb') as file:
			pickle.dump(iterator, file)


class RecordWiseFoldGenerationTask(FoldGenerationTask):
	@override
	def get_splitter(self):
		return StratifiedKFold(
			n_splits=self.n_splits,
			random_state=self.random_state, 
			shuffle=self.shuffle
		)


class SubjectWiseFoldGenerationTask(FoldGenerationTask):
	@override
	def get_splitter(self):
		return StratifiedGroupKFold(
			n_splits=self.n_splits, 
			random_state=self.random_state, 
			shuffle=self.shuffle
		)


class NestedFoldGenerationTask(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	training_config = luigi.DictParameter()

	fold_generation_tasks = {
		'subject_wise': SubjectWiseFoldGenerationTask, 
		'record_wise': RecordWiseFoldGenerationTask
	}

	@override
	def requires(self):
		# Outer fold generation
		outer_technique = self.training_config['outer_fold__technique']
		if outer_technique not in self.fold_generation_tasks.keys():
			raise Exception("Outer fold generation technique not supported.")

		outer_fold_generation_task = self.fold_generation_tasks[outer_technique]
		outer_output_dir = os.path.join(self.output_dir, "nested_cv_folds")
		outer_n_splits = self.training_config['outer_fold__n_splits']
		outer_random_state = self.training_config['outer_fold__random_state']
		outer_shuffle = self.training_config['outer_fold__shuffle']
		metadata_label_column = self.training_config['metadata__label_column']
		metadata_filename = "metadata.csv"

		yield outer_fold_generation_task(
			input_dir=self.input_dir,
			output_dir=outer_output_dir,
			n_splits=outer_n_splits,
			random_state=outer_random_state,
			shuffle=outer_shuffle,
			metadata_label_column=metadata_label_column,
			metadata_filename=metadata_filename
		)

	@override
	def run(self):
		# Inner fold generation
		inner_technique = self.training_config['inner_fold__technique']
		if inner_technique not in self.fold_generation_tasks.keys():
			raise Exception("Inner fold generation technique not supported.")
		
		inner_fold_generation_task = self.fold_generation_tasks[inner_technique]
		inner_n_splits = self.training_config['inner_fold__n_splits']
		inner_random_state = self.training_config['inner_fold__random_state']
		inner_shuffle = self.training_config['inner_fold__shuffle']
		metadata_label_column = self.training_config['metadata__label_column']
		metadata_filename = "train.csv"
		outer_n_splits = self.training_config['outer_fold__n_splits']

		for fold in range(outer_n_splits):
			current_input_dir = os.path.join(self.output_dir, "nested_cv_folds", f"fold_{fold}")
			current_output_dir = os.path.join(self.output_dir, "nested_cv_folds", f"fold_{fold}")
			
			yield inner_fold_generation_task(
				input_dir=current_input_dir,
				output_dir=current_output_dir,
				n_splits=inner_n_splits,
				random_state=inner_random_state,
				shuffle=inner_shuffle,
				metadata_label_column=metadata_label_column,
				metadata_filename=metadata_filename
			)

	@override 
	def output(self):
		output_files = []

		outer_n_splits = self.training_config['outer_fold__n_splits']
		for fold in range(outer_n_splits):
			
			iterator = os.path.join(
				self.output_dir, 
				"nested_cv_folds", 
				f"fold_{fold}", 
				"iterator.pkl"
			)

			output_file = luigi.LocalTarget(iterator)
			output_files.append(output_file)

		return output_files


class FlatFoldGenerationTask(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	training_config = luigi.DictParameter()

	fold_generation_tasks = {
		'subject_wise': SubjectWiseFoldGenerationTask, 
		'record_wise': RecordWiseFoldGenerationTask
	}

	@override
	def run(self):
		# fold generation
		technique = self.training_config['fold__technique']
		if technique not in self.fold_generation_tasks.keys():
			raise Exception("Fold generation technique not supported.")

		fold_generation_task = self.fold_generation_tasks[technique]
		output_dir = os.path.join(self.output_dir, "flat_cv_folds")
		n_splits = self.training_config['fold__n_splits']
		random_state = self.training_config['fold__random_state']
		shuffle = self.training_config['fold__shuffle']
		metadata_label_column = self.training_config['metadata__label_column']
		metadata_filename = self.training_config['metadata__filename']

		yield fold_generation_task(
			input_dir=self.input_dir,
			output_dir=output_dir,
			n_splits=n_splits,
			random_state=random_state,
			shuffle=shuffle,
			metadata_label_column=metadata_label_column,
			metadata_filename=metadata_filename
		)

	@override 
	def output(self):
		outer_n_splits = self.training_config['fold__n_splits']
		iterator = os.path.join(
			self.output_dir, 
			"flat_cv_folds", 
			"iterator.pkl"
		)
		return luigi.LocalTarget(iterator)

