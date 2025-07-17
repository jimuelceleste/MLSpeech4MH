import os 
import pickle 

import luigi

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from typing_extensions import override

from machine_learning.metadata import Metadata
from utility_modules.file import File


class FoldGenerationTask(luigi.Task):
	"""
	Generates folds for cross-validation.

	Parameters
	----------
	input_dir : str
		path to the input directory
	output_dir : str
		path to output directory
	n_splits : int
		Number of train and validate folds to generate
	random_state : int
		Random seed for the fold generation
	shuffle : bool
		True if the samples will be shuffled
	metadata_config : dict
		Metadata configuration

	Methods
	-------
	requires()
		Overrides luigi.Task requires() to require dependencies.
	run()
		Overrides luigi.Task run() to generate folds.
	get_splitter()
		Needs to be overriden in a subclass to return a splitter function.
	save_folds()
		Saves generated folds in the output directory.
	save_iterator()
		Saves the iterator as a pickle file.
	output()
		Overrides luigi.Task output() to define the task outputs.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	n_splits = luigi.IntParameter()
	random_state = luigi.IntParameter()
	shuffle = luigi.BoolParameter()
	metadata_config = luigi.DictParameter()	

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to require dependencies."""
		metadata_file = os.path.join(
			self.input_dir,
			self.metadata_config['file']
		)
		yield File(metadata_file)

	@override
	def run(self):
		"""Overrides luigi.Task run() to generate folds."""
		# Get splitter and the metadata to be splitted
		splitter = self.get_splitter()
		metadata = Metadata(self.input_dir, self.metadata_config)

		# Generate Folds
		record_ids = metadata.get_record_ids()
		subject_ids = metadata.get_subject_ids()
		labels = metadata.get_labels()
		folds = splitter.split(
			X=record_ids, 
			y=labels,
			groups=subject_ids
		)
		folds_list = [(train, validate) for train, validate in folds]

		# Save artifacts
		self.save_folds(folds_list, metadata)
		self.save_iterator(folds_list)

	def get_splitter(self):
		"""Needs to be overriden in a subclass.
		
		Returns the splitter function. 
		"""
		pass

	def save_folds(self, folds, metadata):
		"""Saves generated folds in the output directory.
		
		The folds are saved in directories named K_fold, 
		where K is the fold number, starting from 0.

		Parameters
		----------
		folds : list
			the generated folds
		metadata : Metadata
			the metadata instance 

		Returns 
		-------
		None
		"""
		for fold, (train, validate) in enumerate(folds):	
			train_df = metadata.get_rows_by_index(train)
			validate_df = metadata.get_rows_by_index(validate)
			
			current_output_dir = os.path.join(self.output_dir, f"fold_{fold}")
			train_file = os.path.join(current_output_dir, "train.csv")
			validate_file = os.path.join(current_output_dir, "validate.csv")

			os.makedirs(current_output_dir, exist_ok=True)
			train_df.to_csv(train_file, index=False)
			validate_df.to_csv(validate_file, index=False)

		return None 

	def save_iterator(self, iterator):
		"""Saves the iterator as a pickle file.

		Parameters
		----------
		iterator : sklearn._BaseKFold

		Returns 
		-------
		None
		"""
		iterator_file = os.path.join(self.output_dir, "iterator.pkl")
		with open(iterator_file, 'wb') as file:
			pickle.dump(iterator, file)
		
		return None

	@override
	def output(self):
		"""Overrides luigi.Task output() to define the task outputs."""
		# For all folds: train.csv and validate.csv
		output_files = []
		for fold in range(self.n_splits):
			train_file = os.path.join(self.output_dir, f"fold_{fold}", "train.csv")
			validate_file = os.path.join(self.output_dir, f"fold_{fold}", "validate.csv")
			
			output_files.append(luigi.LocalTarget(train_file))
			output_files.append(luigi.LocalTarget(validate_file))

		# Iterator file: iterator.pkl
		iterator_file = os.path.join(self.output_dir, "iterator.pkl")
		output_files.append(luigi.LocalTarget(iterator_file))

		return output_files


class RecordWiseFoldGenerationTask(FoldGenerationTask):
	"""
	Implements record-wise fold generation.

	Read more about this protocol at: https://doi.org/10.1093/gigascience/gix019

	Methods
	-------
	get_splitter()
		Returns an instance of sklearn.StratifiedKFold.
	"""
	@override
	def get_splitter(self):
		"""Returns an instance of sklearn.StratifiedKFold."""
		return StratifiedKFold(
			n_splits=self.n_splits,
			random_state=self.random_state, 
			shuffle=self.shuffle
		)


class SubjectWiseFoldGenerationTask(FoldGenerationTask):
	"""
	Implements subject-wise fold generation.

	Read more about this protocol at: https://doi.org/10.1093/gigascience/gix019

	Methods
	-------
	get_splitter()
		Returns an instance of sklearn.StratifiedGroupKFold.
	"""
	@override
	def get_splitter(self):
		"""Returns an instance of sklearn.StratifiedGroupKFold."""
		return StratifiedGroupKFold(
			n_splits=self.n_splits, 
			random_state=self.random_state, 
			shuffle=self.shuffle
		)


class NestedFoldGenerationTask(luigi.Task):
	"""
	Generates folds for nested cross-validation.

	Parameters
	----------
	input_dir : str 
		path to the input directory
	output_dir : str 
		path to the output directory
	metadata_config : dict
		metadata configuration
	training_config : dict
		training configuration

	Attributes
	----------
	_fold_generation_tasks : dict 
		dictionary of supported fold generation tasks
	
	Methods 
	-------
	requires()
		Overrides luigi.Task requires() to require dependencies
	run()
		Overrides luigi.task run() to generate folds for cross-validation.
	output()
		Overrides luigi.Task output() to define the task outputs.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()

	_fold_generation_tasks = {
		'subject_wise': SubjectWiseFoldGenerationTask, 
		'record_wise': RecordWiseFoldGenerationTask
	}

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to require dependencies.
		
		Outer folds are generated here.
		"""
		# Outer fold generation
		technique = self.training_config['outer_fold__technique']
		if technique not in self._fold_generation_tasks.keys():
			raise Exception("Outer fold generation technique not supported.")

		fold_generation_task = self._fold_generation_tasks[technique]
		yield fold_generation_task(
			input_dir=self.input_dir,
			output_dir=os.path.join(self.output_dir, "nested_cv_folds"),
			n_splits=self.training_config['outer_fold__n_splits'],
			random_state=self.training_config['outer_fold__random_state'],
			shuffle=self.training_config['outer_fold__shuffle'],
			metadata_config=self.metadata_config
		)

	@override
	def run(self):
		"""Overrides luigi.task run() to generate folds for cross-validation.

		Inner folds are generated here.
		"""
		# Inner fold generation
		technique = self.training_config['inner_fold__technique']
		if technique not in self._fold_generation_tasks.keys():
			raise Exception("Inner fold generation technique not supported.")
		
		fold_generation_task = self._fold_generation_tasks[technique]
		outer_n_splits = self.training_config['outer_fold__n_splits']
		
		# Splitting the train.csv for inner folds
		new_metadata_config = dict(self.metadata_config).copy()
		new_metadata_config['file'] = "train.csv"
		
		for fold in range(outer_n_splits):
			current_dir = os.path.join(
				self.output_dir, 
				"nested_cv_folds", 
				f"fold_{fold}"
			)
			
			yield fold_generation_task(
				input_dir=current_dir,
				output_dir=current_dir,
				n_splits=self.training_config['inner_fold__n_splits'],
				random_state=self.training_config['inner_fold__random_state'],
				shuffle=self.training_config['inner_fold__shuffle'],
				metadata_config=new_metadata_config
			)

	@override 
	def output(self):
		"""Overrides luigi.Task output() to define the task outputs."""
		output_files = []

		outer_n_splits = self.training_config['outer_fold__n_splits']
		for fold in range(outer_n_splits):
			current_dir = os.path.join(
				self.output_dir, 
				"nested_cv_folds", 
				f"fold_{fold}"
			)
			iterator = os.path.join(current_dir, 'iterator.pkl')
			output_file = luigi.LocalTarget(iterator)
			output_files.append(output_file)

		return output_files


class FlatFoldGenerationTask(luigi.Task):
	"""
	Generates folds for flat cross-validation.

	Parameters
	----------
	input_dir : str 
		path to the input directory
	output_dir : str 
		path to the output directory
	metadata_config : dict
		metadata configuration
	training_config : dict
		training configuration

	Attributes
	----------
	_fold_generation_tasks : dict 
		dictionary of supported fold generation tasks
	
	Methods 
	-------
	requires()
		Overrides luigi.Task requires() to require dependencies.
	run()
		Overrides luigi.task run() to generate folds for cross-validation.
	output()
		Overrides luigi.Task output() to define the task outputs.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()

	_fold_generation_tasks = {
		'subject_wise': SubjectWiseFoldGenerationTask, 
		'record_wise': RecordWiseFoldGenerationTask
	}

	@override
	def run(self):
		"""Overrides luigi.task run() to generate folds for cross-validation."""
		technique = self.training_config['fold__technique']
		if technique not in self._fold_generation_tasks.keys():
			raise Exception("Fold generation technique not supported.")

		fold_generation_task = self._fold_generation_tasks[technique]
		output_dir = os.path.join(self.output_dir, "flat_cv_folds")
		
		yield fold_generation_task(
			input_dir=self.input_dir,
			output_dir=output_dir,
			n_splits=self.training_config['fold__n_splits'],
			random_state=self.training_config['fold__random_state'],
			shuffle=self.training_config['fold__shuffle'],
			metadata_config=self.metadata_config
		)

	@override 
	def output(self):
		"""Overrides luigi.Task output() to define the task outputs."""
		iterator = os.path.join(
			self.output_dir, 
			"flat_cv_folds", 
			"iterator.pkl"
		)
		return luigi.LocalTarget(iterator)