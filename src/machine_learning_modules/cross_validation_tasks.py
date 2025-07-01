import pickle
import os

import luigi
import pandas as pd
from typing_extensions import override

from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import root_mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_percentage_error

from machine_learning_modules.fold_generation_tasks import NestedFoldGenerationTask
from machine_learning_modules.fold_generation_tasks import FlatFoldGenerationTask
from machine_learning_modules.metadata import Metadata
from machine_learning_modules.optimization import HyperparameterOptimizationCVTask
from utility_modules.file import File

class NestedCrossValidationTask(luigi.Task):
	"""
	Runs nested cross validation protocol on the specified pipeline. 
	
	Parameters:
		input_dir (str): Path to the input directory.
		output_dir (str): Path to the output directory.
		training_config (dict): Training parameters
		pipeline_config (dict): Pipeline configuration: steps and parameters
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()

	evaluation_metrics = {
		# Classification Metrics
		"accuracy_score": accuracy_score, 
		"f1_score": f1_score, 
		"precision_score": precision_score,
		"recall_score": recall_score, 
		"specificity_score": specificity_score,
		"sensitivity_score": sensitivity_score,
		"geometric_mean_score": geometric_mean_score,
		"recall_macro_score": recall_score,
		"roc_auc_score": roc_auc_score,
		
		# Regression Metrics
		"mean_absolute_error": mean_absolute_error,
		"root_mean_squared_error": root_mean_squared_error,
		"r2_score": r2_score,
		"mean_squared_error": mean_squared_error,
		"mean_absolute_percentage_error": mean_absolute_percentage_error
	}

	@override
	def requires(self):
		# Metadata
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		yield File(metadata_file)

		# Features
		for input_file in self._input_iterator():
			yield File(input_file)

		# Generate Folds
		yield NestedFoldGenerationTask(
			input_dir=self.input_dir,
			output_dir=self.output_dir,
			training_config=self.training_config
		)

	@override
	def run(self):
	
		# Get Folds
		outer_fold_iterator = self.load_outer_fold_iterator()
		outer_metadata = self.load_metadata()
		performance = {}

		for fold, (train, validate) in enumerate(outer_fold_iterator):
			
			current_fold_iterator_file = os.path.join(
				self.output_dir, 
				"nested_cv_folds",
				f"fold_{fold}",
				"iterator.pkl"
			)

			current_metadata_file = os.path.join(
				self.output_dir,
				"nested_cv_folds",
				f"fold_{fold}",
				"train.csv"
			)

			current_output_dir = os.path.join(
				self.output_dir,
				self.pipeline_config['unique_id'], 
				f"fold_{fold}"
			)

			yield HyperparameterOptimizationCVTask(
				input_dir=self.input_dir,
				output_dir=current_output_dir,
				training_config=self.training_config,
				pipeline_config=self.pipeline_config,
				fold_iterator_file=current_fold_iterator_file,
				metadata_file=current_metadata_file
			)

			model = self.load_model(current_output_dir)
			validate_features = self.load_features(outer_metadata, validate)
			validate_labels = self.load_labels(outer_metadata, validate)
			validate_predictions = model.predict(validate_features)
			validate_performance = self.get_performance(validate_predictions, validate_labels)
			performance[fold] = validate_performance
			validate_filenames = outer_metadata.get_filenames_by_index(validate)
			self.save_validate_predictions(
				metadata=outer_metadata,
				validate_index=validate,
				predictions=validate_predictions, 
				labels=validate_labels,
				fold=fold
			)

		outer_iterator_file = os.path.join(
			self.output_dir, 
			"nested_cv_folds", 
			"iterator.pkl"
		)

		outer_output_dir = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'], 
			"final_model"
		)

		metadata_file = os.path.join(
			self.input_dir, 
			"metadata.csv"
		)

		yield HyperparameterOptimizationCVTask(
			input_dir=self.input_dir,
			output_dir=outer_output_dir,
			training_config=self.training_config,
			pipeline_config=self.pipeline_config,
			fold_iterator_file=outer_iterator_file,
			metadata_file=metadata_file
		)

		# Save performance
		self.save_performance(performance)

	@override
	def output(self):
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		dependencies = [luigi.LocalTarget(performance_file)]

		outer_n_splits = self.training_config['outer_fold__n_splits']
		for fold in range(outer_n_splits):
			prediction_file = os.path.join(
				self.output_dir, 
				self.pipeline_config['unique_id'],
				f"fold_{fold}",
				"predictions.csv"
			)
			dependencies.append(luigi.LocalTarget(prediction_file))

		return dependencies
	
	def _input_iterator(self):
		metadata = self.load_metadata()
	
		for file in metadata.get_filenames():
			input_file = os.path.join(self.input_dir, file)
	
			yield input_file

	def load_metadata(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")		
		label_column = self.training_config['metadata__label_column']
		metadata = Metadata(metadata_file, label_column)		
		return metadata

	def load_pickle(self, file):
		with open(file, 'rb') as file:
			return pickle.load(file)

	def load_outer_fold_iterator(self):
		iterator_file = os.path.join(
			self.output_dir, 
			"nested_cv_folds", 
			"iterator.pkl"
		)
		iterator = self.load_pickle(iterator_file)
		return iterator

	def load_features(self, metadata, index):
		filenames = metadata.get_filenames_by_index(index)

		features_list = []
		for file in filenames:
			features_file = os.path.join(self.input_dir, file)
			features = pd.read_csv(features_file)
			features_list.append(features)	
		
		features_list = pd.concat(features_list)#.values.tolist()
		return features_list

	def load_labels(self, metadata, index):
		labels = metadata.get_labels_by_index(index)
		return labels.values.tolist()

	def load_inner_fold_iterator(self, fold):
		iterator_file = os.path.join(
			self.output_dir, 
			"nested_cv_folds", 
			f"fold_{fold}",
			"iterator.pkl"
		)
		iterator = self.load_pickle(iterator_file)
		return iterator

	def load_model(self, current_dir):
		model_file = os.path.join(
			current_dir,
			"model.pkl"
		)
		model = self.load_pickle(model_file)
		return model

	def get_performance(self, predictions, labels):
		evaluation_metrics = self.training_config['evaluation__metrics']
		performance = {}
		for evaluation_metric in evaluation_metrics:
			# Check-point: is the evaluation metric supported?
			if evaluation_metric not in self.evaluation_metrics.keys():
				raise Exception(f"Evaluation metric not supported: {evaluation_metric}")
			
			current_function = self.evaluation_metrics[evaluation_metric]
			
			# Modify me for a cleaner execution
			if evaluation_metric == "recall_macro_score": # Unbalanced Average Recall
				performance[evaluation_metric] = current_function(
					y_true=labels, 
					y_pred=predictions,
					average='macro'
				)
			else:
				performance[evaluation_metric] = current_function(y_true=labels, y_pred=predictions)
		return performance

	def save_validate_predictions(self, metadata, validate_index, predictions, labels, fold):
		output_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'], 
			f"fold_{fold}",
			"predictions.csv"
		)
		predictions_df = pd.DataFrame.from_dict({
			'filename': metadata.get_filenames_by_index(validate_index).values.tolist(),
			'predictions': predictions,
			'labels': labels
		})
		predictions_df.to_csv(output_file, index=False)

	def save_performance(self, performance):
		performance = pd.DataFrame.from_dict(performance)
		performance['average'] = performance.mean(axis=1)
		performance['std'] = performance.std(axis=1)
		performance = performance.T
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		performance.to_csv(performance_file)


class FlatCrossValidationTask(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()

	def requires(self):
		# Metadata
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		yield File(metadata_file)

		# Features
		for input_file in self._input_iterator():
			yield File(input_file)

		# Generate Folds
		yield FlatFoldGenerationTask(
			input_dir=self.input_dir,
			output_dir=self.output_dir,
			training_config=self.training_config
		)

	def run(self):
		# Get Folds
		fold_iterator = self.load_fold_iterator()
		outer_metadata = self.load_metadata()
		performance = {}

		fold_iterator_file = os.path.join(
			self.output_dir, 
			"flat_cv_folds",
			"iterator.pkl"
		)

		metadata_file = os.path.join(
			self.input_dir,
			"metadata.csv"
		)

		output_dir = os.path.join(
			self.output_dir,
			self.pipeline_config['unique_id']
		)

		yield HyperparameterOptimizationCVTask(
			input_dir=self.input_dir,
			output_dir=output_dir,
			training_config=self.training_config,
			pipeline_config=self.pipeline_config,
			fold_iterator_file=fold_iterator_file,
			metadata_file=metadata_file
		)

	def output(self):
		performance_file = os.path.join(
			self.output_dir, 
			self.pipeline_config['unique_id'],
			"performance.csv"
		)
		return luigi.LocalTarget(performance_file)

	def _input_iterator(self):
		metadata = self.load_metadata()
		for file in metadata.get_filenames():
			input_file = os.path.join(self.input_dir, file)
			yield input_file
	
	def load_fold_iterator(self):
		iterator_file = os.path.join(
			self.output_dir, 
			"flat_cv_folds", 
			"iterator.pkl"
		)
		iterator = self.load_pickle(iterator_file)
		return iterator

	def load_metadata(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")		
		label_column = self.training_config['metadata__label_column']
		metadata = Metadata(metadata_file, label_column)		
		return metadata

	def load_pickle(self, file):
		with open(file, 'rb') as file:
			return pickle.load(file)