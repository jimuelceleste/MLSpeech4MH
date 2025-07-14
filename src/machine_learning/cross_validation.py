import os
import pickle

import luigi
import pandas as pd

from typing_extensions import override

from machine_learning.fold_generation import NestedFoldGenerationTask
from machine_learning.fold_generation import FlatFoldGenerationTask
from machine_learning.hyperparameter_optimization import HyperparameterOptimizationCVTask
from machine_learning.metadata import Metadata
from machine_learning.model_evaluation import ModelEvaluationTask
from utility_modules.file import File

class NestedCrossValidationTask(luigi.Task):
	"""
	Runs the Nested Cross-Validation Protocol.

	Read more about this protocol: https://doi.org/10.1016/j.eswa.2021.115222
	
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
	pipeline_config : dict
		pipeline configuration
	
	Methods
    -------
	requires()
		Overrides luigi.Task requires() to require dependencies.
	run()
		Implements nested cross-validation.
	generate_summary()
		Generates performance summary.
	output()
		Defines task output.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to require dependencies."""
		
		# Metadata file
		metadata_file = os.path.join(
			self.input_dir, 
			self.metadata_config['file']
		)
		yield File(metadata_file)

		# Features enumerated in metadata
		metadata = Metadata(self.input_dir, self.metadata_config)
		for file in metadata.get_filenames():
			input_file = os.path.join(self.input_dir, file)
			yield File(input_file)

		# Cross-Validation Folds
		yield NestedFoldGenerationTask(
			input_dir=self.input_dir,
			output_dir=self.output_dir,
			metadata_config=self.metadata_config,
			training_config=self.training_config
		)

	@override
	def run(self):
		"""Implements nested cross-validation."""

		# 1/3 NESTED CROSS-VALIDATION

		# Training metadata configuration
		train_metadata_config = dict(self.metadata_config).copy()
		train_metadata_config['file'] = 'train.csv'

		# Validate metadata configuration
		validate_metadata_config = dict(self.metadata_config).copy()
		validate_metadata_config['file'] = 'validate.csv'

		# For all folds, train and evaluate model
		n_splits = self.training_config['outer_fold__n_splits']
		for fold in range(n_splits):
			current_output_dir = os.path.join(
				self.output_dir,
				self.pipeline_config['unique_id'], 
				f"fold_{fold}"
			)
			current_metadata_dir = os.path.join(
				self.output_dir,
				"nested_cv_folds",
				f"fold_{fold}"
			)
			current_fold_iterator_file = os.path.join(
				current_metadata_dir,
				'iterator.pkl'
			)

			yield HyperparameterOptimizationCVTask(
				input_dir=self.input_dir,
				output_dir=current_output_dir,
				metadata_dir=current_metadata_dir,
				metadata_config=train_metadata_config,
				training_config=self.training_config,
				pipeline_config=self.pipeline_config,
				fold_iterator_file=current_fold_iterator_file
			)

			yield ModelEvaluationTask(
				input_dir=self.input_dir,
				output_dir=current_output_dir,
				model_dir=current_output_dir,
				metadata_dir=current_metadata_dir,
				metadata_config=validate_metadata_config,
				training_config=self.training_config
			)

		# 2/3 TRAINING FINAL MODEL
		final_model_output_dir = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'], 
			"final_model"
		)

		final_model_fold_iterator_file = os.path.join(
			self.output_dir, 
			"nested_cv_folds", 
			"iterator.pkl"
		)

		yield HyperparameterOptimizationCVTask(
			input_dir=self.input_dir,
			output_dir=final_model_output_dir,
			metadata_dir=self.input_dir,
			metadata_config=self.metadata_config,
			training_config=self.training_config,
			pipeline_config=self.pipeline_config,
			fold_iterator_file=final_model_fold_iterator_file
		)

		# 3/3 GENERATING SUMMARY
		self.generate_summary()

	def generate_summary(self):
		"""Generates performance summary."""
		n_splits = self.training_config['outer_fold__n_splits']
		performance = []

		for fold in range(n_splits):
			fold_performance_file = os.path.join(
				self.output_dir,
				self.pipeline_config['unique_id'],
				f"fold_{fold}",
				"outer_validation_performance.csv"
			)
			fold_performance = pd.read_csv(fold_performance_file)
			performance.append(fold_performance)

		performance = pd.concat(performance).reset_index(drop=True)
		performance.loc['mean'] = performance.mean(axis=0)
		performance.loc['std'] = performance.std(axis=0)
		
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		performance.to_csv(performance_file, index=True)

	@override
	def output(self):
		"""Defines task output."""
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		return luigi.LocalTarget(performance_file)


class FlatCrossValidationTask(luigi.Task):
	"""
	Runs the Flat Cross-Validation Protocol.

	Read more about this protocol: https://doi.org/10.1016/j.eswa.2021.115222
	
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
	pipeline_config : dict
		pipeline configuration
	
	Methods
    -------
	requires()
		Overrides luigi.Task requires() to require dependencies.
	run()
		Implements flat cross-validation.
	output()
		Defines task output.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to require dependencies."""
		
		# Metadata file
		metadata_file = os.path.join(
			self.input_dir, 
			self.metadata_config['filename']
		)
		yield File(metadata_file)

		# Features enumerated in metadata
		metadata = Metadata(self.input_dir, self.metadata_config)
		for file in metadata.get_filenames():
			input_file = os.path.join(self.input_dir, file)
			yield File(input_file)

		# Cross-Validation Folds
		yield FlatFoldGenerationTask(
			input_dir=self.input_dir,
			output_dir=self.output_dir,
			metadata_config=self.metadata_config,
			training_config=self.training_config
		)	

	def run(self):
		"""Implements flat cross-validation."""
		current_output_dir = os.path.join(
			self.output_dir,
			self.pipeline_config['unique_id']
		)
		
		fold_iterator_file = os.path.join(
			self.output_dir,
			"flat_cv_folds",
			"iterator.pkl"
		)

		yield HyperparameterOptimizationCVTask(
			input_dir=self.input_dir,
			output_dir=current_output_dir,
			metadata_dir=self.input_dir,
			metadata_config=self.metadata_config,
			training_config=self.training_config,
			pipeline_config=self.pipeline_config,
			fold_iterator_file=fold_iterator_file
		)

		yield ModelEvaluationTask(
			input_dir=self.input_dir,
			output_dir=current_output_dir,
			model_dir=current_output_dir,
			metadata_dir=self.input_dir,
			metadata_config=self.metadata_config,
			training_config=self.training_config
		)

	def output(self):
		"""Defines task output."""
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		return luigi.LocalTarget(performance_file)
