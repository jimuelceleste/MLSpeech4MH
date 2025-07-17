import os
import pickle

import luigi
import imblearn
import pandas as pd
import sklearn 

from typing_extensions import override

from machine_learning.metadata import Metadata
from utility_modules.file import File


def uar_score(y_true, y_pred):
	"""Unweighted Average Recall (UAR)
	
	Parameters
	----------
	y_true : list 
		list of ground truth labels
	y_pred : list
		list of predictions

	Returns
	-------
	float 
		UAR score
	"""
	return sklearn.metrics.recall_score(
		y_true=y_true, 
		y_pred=y_pred,
		average='macro'
	)


class ModelEvaluationTask(luigi.Task):
	"""Implements model evaluation.

	Parameters
	----------
	input_dir : str 
		path to the input directory (features)
	output_dir : str 
		path to the output directory
	model_dir : str 
		path to the model directory (model should be named model.pkl)
	metadata_dir : str 
		path to the metadata directory 
	metadata_config : dict 
		metadata configuration 
	training_config : dict 
		training configuration
	
	Attributes
	----------
	_classification_evaluation_metrics : dict
		dictionary of supported evaluation metrics for classification
	_regression_evaluation_metrics : dict 
		dictionary of supported evaluation metrics for regression

	Methods
	-------
	requires()
		Overrides luigi.Task requires() to require task dependencies.
	run()
		Implements model evaluation.
	get_model()
		Returns model instance loaded with pickle.
	get_performance()
		Calculates the performance of the model from predictions and labels.
	save_predictions(metadata, predictions, labels)
		Saves model predictions.
	save_performance(performance)
		Saves outer validation performance.
	output()
		Overrides luigi.Task output() to define task outputs.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	model_dir = luigi.Parameter()
	metadata_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()
	
	_classification_evaluation_metrics = {
		"accuracy_score": sklearn.metrics.accuracy_score, 
		"f1_score": sklearn.metrics.f1_score, 
		"precision_score": sklearn.metrics.precision_score,
		"recall_score": sklearn.metrics.recall_score, 
		"roc_auc_score": sklearn.metrics.roc_auc_score,
		"specificity_score": imblearn.metrics.specificity_score,
		"sensitivity_score": imblearn.metrics.sensitivity_score, 
		"uar_score": uar_score,
		# IMPORTANT: MODIFY UAR TO GET MACRO RECALL
	}

	_regression_evaluation_metrics = {
		"mean_absolute_error": sklearn.metrics.mean_absolute_error,
		"mean_squared_error": sklearn.metrics.mean_squared_error,
		"root_mean_squared_error": sklearn.metrics.root_mean_squared_error,
		"r2_score": sklearn.metrics.r2_score,
	}

	_evaluation_metrics = {
		**_classification_evaluation_metrics,
		**_regression_evaluation_metrics,
	}

	def requires(self):
		"""Overrides luigi.Task requires() to require task dependencies."""
		model_file = os.path.join(self.model_dir, "model.pkl")
		yield File(model_file)

	@override
	def run(self):
		"""Implements model evaluation."""
		metadata = Metadata(self.metadata_dir, self.metadata_config)
		validate_features = metadata.get_features(self.input_dir)
		validate_labels = metadata.get_labels().values.tolist()

		model = self.get_model()
		predictions = model.predict(validate_features)
		performance = self.get_performance(
			labels=validate_labels, 
			predictions=predictions
		)

		self.save_predictions(
			metadata=metadata,
			predictions=predictions, 
			labels=validate_labels,
		)
		self.save_performance(performance)

	def get_model(self):
		"""Returns model instance loaded with pickle."""
		model_file = os.path.join(self.model_dir, "model.pkl")
		with open(model_file, 'rb') as file:
			return pickle.load(file)

	def get_performance(self, labels, predictions):
		"""
		Calculates the performance of the model from predictions and labels.
		
		Parameters
		----------
		predictions : list 
			list of predictions 
		labels : list 
			list of ground truth labels (same order as predictions)
		"""
		evaluation_metrics = self.training_config['evaluation__metrics']
		performance = {}
		for metric in evaluation_metrics:
			# Check-point: is the evaluation metric supported?
			if metric not in self._evaluation_metrics.keys():
				raise Exception(f"Evaluation metric not supported: {metric}")
			current_metric = self._evaluation_metrics[metric]
			performance[metric] = current_metric(y_true=labels, y_pred=predictions)
		return performance

	def save_predictions(self, metadata, predictions, labels):
		"""Saves model predictions.
		
		Parameters 
		----------
		metadata : object 
			Metadata instance 
		predictions : list 
			list of predictions 
		labels : list 
			list of labels
		"""
		output_file = os.path.join(
			self.output_dir, 
			"outer_validation_predictions.csv"
		)
		predictions_df = pd.DataFrame.from_dict({
			'filename': metadata.get_filenames().values.tolist(),
			'predictions': predictions,
			'labels': labels
		})
		predictions_df.to_csv(output_file, index=False)

	def save_performance(self, performance):
		"""Saves outer validation performance.
		
		Parameters
		----------
		performance : dict 
			dictionary of {metric : str, value : float)} pairs
		"""
		performance = pd.DataFrame([performance])
		performance_file = os.path.join(
			self.output_dir, 
			"outer_validation_performance.csv"
		)
		performance.to_csv(performance_file, index=False)
	
	@override
	def output(self):
		"""Overrides luigi.Task output() to define task outputs."""
		performance_file = os.path.join(
			self.output_dir, 
			"outer_validation_performance.csv"
		)

		prediction_file = os.path.join(
			self.output_dir, 
			"outer_validation_predictions.csv"
		)

		dependencies = [
			luigi.LocalTarget(performance_file),
			luigi.LocalTarget(prediction_file)
		]

		return dependencies


class NestedModelEvaluationTask(ModelEvaluationTask):
	pass


class FlatModelEvaluationTask(ModelEvaluationTask):
	@override
	def save_predictions(self, metadata, predictions, labels):
		"""Saves model predictions.
		
		Parameters 
		----------
		metadata : object 
			Metadata instance 
		predictions : list 
			list of predictions 
		labels : list 
			list of labels
		"""
		output_file = os.path.join(
			self.output_dir, 
			"validation_predictions.csv"
		)
		predictions_df = pd.DataFrame.from_dict({
			'filename': metadata.get_filenames().values.tolist(),
			'predictions': predictions,
			'labels': labels
		})
		predictions_df.to_csv(output_file, index=False)

	@override
	def save_performance(self, performance):
		"""Saves outer validation performance.
		
		Parameters
		----------
		performance : dict 
			dictionary of {metric : str, value : float)} pairs
		"""
		performance = pd.DataFrame([performance])
		performance_file = os.path.join(
			self.output_dir, 
			"validation_performance.csv"
		)
		performance.to_csv(performance_file, index=False)
	
	@override
	def output(self):
		"""Overrides luigi.Task output() to define task outputs."""
		performance_file = os.path.join(
			self.output_dir, 
			"validation_performance.csv"
		)

		prediction_file = os.path.join(
			self.output_dir, 
			"validation_predictions.csv"
		)

		dependencies = [
			luigi.LocalTarget(performance_file),
			luigi.LocalTarget(prediction_file)
		]

		return dependencies