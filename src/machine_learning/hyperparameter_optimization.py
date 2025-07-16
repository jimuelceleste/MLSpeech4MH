import os 
import pickle 

import imblearn
import luigi
import pandas as pd
import skopt
import sklearn 
import xgboost

from imblearn.pipeline import Pipeline
from typing_extensions import override

from machine_learning.metadata import Metadata
from utility_modules.file import File

class HyperparameterOptimizationCVTask(luigi.Task):
	"""
	Implements hyperparameter optimization.

	Supported tasks: 
	1. GridSearch 
	2. BayesSearch
	
	Parameters
	----------
	input_dir : str 
		path to the features directory
	output_dir : str
		path to the output directory
	metadata_dir : str 
		path to the metadata directory
	metadata_config : dict 
		metadata configuration 
	training_config : dict
		training configuration
	pipeline_config : dict 
		pipeline configuration
	fold_iterator_file : str 
		path to the fold iterator file

	Attributes
	----------
	_optimization_techniques : dict
		dictionary of supported optimization techniques
	_feature_scaling_techniques : dict 
		dictionary of supported feature scaling techniques 
	_feature_selection_techniques : dict 
		dictionary of supported feature selection techniques
	_dimensionality_reduction_techniques : dict 
		dictionary of supported dimensionality reduction techniques
	_data_augmentation_techniques : dict 
		dictionary of supported data augmentation techniques 
	_classification_models : dict 
		dictionary of supported classification models 
	_regression_models : dict 
		dictionary of supported regression models
	_supported_tasks : dict 
		dictionary of all supported tasks

	Methods
	-------
	requires()
		Overrides luigi.Task requires() to require dependencies.
	run()
		Implements hyperparameter optimization.
	get_pipeline()
		Returns the pipeline instance.
	get_folds()
		Returns the fold iterator.
	get_optimizer(pipeline, param_grid, cv, scoring)
		Returns the optimizer class.
	save_best_model(model)
		Saves the best model.
	save_hyperparameters(hyperparameters)
		Saves the best hyperparameters.
	save_performance(performance)
		Saves the performance on all fits.
	save_features(feature_names)
		Saves the list of final set of features.
	save_best_performance(performance, best_index)
		Saves hyperparameter optmization information for the best model.
	output()
		Overrides luigi.Task output() to define task outputs.
	"""

	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	metadata_dir = luigi.Parameter()
	metadata_config = luigi.DictParameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()	
	fold_iterator_file = luigi.Parameter()

	_optimization_techniques = {
		'grid_search': sklearn.model_selection.GridSearchCV,
		'bayes_search': skopt.BayesSearchCV
	}
	_feature_scaling_techniques = {
		'standard_scaler': sklearn.preprocessing.StandardScaler,
	}
	_feature_selection_techniques = {
		'select_k_best': sklearn.feature_selection.SelectKBest,
	}
	_dimensionality_reduction_techniques = {
		'pca': sklearn.decomposition.PCA, 
		'lsa': sklearn.decomposition.TruncatedSVD,
	}
	_data_augmentation_techniques = {
		'smote': imblearn.over_sampling.SMOTE
	}
	_classification_models = {
		'random_forest_classifier': sklearn.ensemble.RandomForestClassifier,
		'support_vector_classifier': sklearn.svm.SVC, 
		'xgb_classifier': xgboost.XGBClassifier,
		'logistic_regression_classifier': sklearn.linear_model.LogisticRegression,
		'multilayer_perceptron_classifier': sklearn.neural_network.MLPClassifier,
		'naive_bayes_classifier': sklearn.naive_bayes.GaussianNB,
	}
	_regression_models = {
		'random_forest_regressor': sklearn.ensemble.RandomForestRegressor,
		'support_vector_regressor': sklearn.svm.SVR,
		'xgb_regressor': xgboost.XGBRegressor, 
		'linear_regression': sklearn.linear_model.LinearRegression, 
		'lasso_regressor': sklearn.linear_model.Lasso,
		'elastic_net_regressor': sklearn.linear_model.ElasticNet,
		'multilayer_perceptron_regressor': sklearn.neural_network.MLPRegressor
	}

	_supported_tasks = {
		**_optimization_techniques,
		**_feature_scaling_techniques,
		**_feature_selection_techniques,
		**_dimensionality_reduction_techniques,
		**_data_augmentation_techniques,
		**_classification_models,
		**_regression_models
	}

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to require dependencies.
		1. Metadata file 
		2. Fold iterator 
		3. Features
		"""
		# Metadata
		metadata_file = os.path.join(
			self.metadata_dir, 
			self.metadata_config['file']
		)
		yield File(metadata_file)

		# Fold iterator
		yield File(self.fold_iterator_file)

		# Features
		metadata = Metadata(self.metadata_dir, self.metadata_config)
		for file in metadata.get_filenames():
			input_file = os.path.join(self.input_dir, file)
			yield File(input_file)
	
	@override 
	def run(self):
		"""Implements hyperparameter optimization."""
		# Set-up optimization task 
		optimizer = self.get_optimizer(
			pipeline=self.get_pipeline(), 
			param_grid=dict(self.pipeline_config['parameters']), 
			cv=self.get_folds(),
			scoring=self.training_config['optimization__scoring'],
		)

		# Get training data
		metadata = Metadata(self.metadata_dir, self.metadata_config)
		train_features = metadata.get_features(self.input_dir)
		train_labels = metadata.get_labels().values.tolist()

		# Run optimization task
		optimizer.fit(train_features, train_labels)
		
		# Save artifacts
		best_model = optimizer.best_estimator_.fit(train_features, train_labels)
		best_hyperparameters = optimizer.best_params_
		performance = optimizer.cv_results_
		best_index = optimizer.best_index_.T
		features = best_model[:-1].get_feature_names_out()

		os.makedirs(self.output_dir, exist_ok=True)
		self.save_best_model(best_model)
		self.save_best_hyperparameters(best_hyperparameters)
		self.save_best_performance(performance, best_index)
		self.save_performance(performance)
		self.save_features(features)

	def get_pipeline(self):
		"""Returns the pipeline instance."""
		steps = self.pipeline_config['steps']
		pipeline = []
		for step in steps:
			task = steps[step]
			if task not in self._supported_tasks:
				raise Exception(f'Machine learning task not supported: {task}')
			task_instance = self._supported_tasks[task]()
			pipeline.append((step, task_instance))
		return Pipeline(pipeline)

	def get_folds(self):
		"""Returns the fold iterator."""
		with open(self.fold_iterator_file, 'rb') as file:
			folds = pickle.load(file)
		return folds 

	def get_optimizer(self, pipeline, param_grid, cv, scoring):
		"""Returns the optimizer class.
		
		Supported optimizers:
		1. Grid Search
		2. Bayes Search 
		"""
		optimization_technique = self.training_config['optimization__technique']
		if optimization_technique not in self._optimization_techniques.keys():
			raise Exception('Optimization technique not supported.')
		
		optimizer = self._optimization_techniques[optimization_technique](
			pipeline,
			param_grid,
			cv=cv,
			scoring=scoring,
			n_jobs=-1, 
			return_train_score=True,
			verbose=True
		)

		return optimizer

	def save_best_model(self, model):
		"""Saves the best model."""
		file = os.path.join(self.output_dir, "model.pkl")
		
		with open(file, 'wb') as f:
			pickle.dump(model, f)

		return None

	def save_best_hyperparameters(self, hyperparameters):
		"""Saves the best hyperparameters."""
		file = os.path.join(self.output_dir, "hyperparameters.csv")
	
		for key in hyperparameters:
			val = hyperparameters[key]
			if isinstance(val, tuple) or isinstance(val, list):
				hyperparameters[key] = " ".join(str(x) for x in val)

		hyp_df = pd.DataFrame(hyperparameters, index=[0])
		hyp_df.to_csv(file, index=False)

		return None 

	def save_best_performance(self, performance, best_index):
		"""Saves hyperparameter optmization information for the best model."""
		performance_file = os.path.join(self.output_dir, "best_cv_performance.csv")
		performance_df = pd.DataFrame.from_dict(performance)
		best_performance = performance_df.iloc[best_index]
		best_performance.to_csv(performance_file, index=True)

	def save_performance(self, performance):
		"""Saves the performance on all fits."""
		performance_file = os.path.join(self.output_dir, "cv_performance.csv")
		performance_df = pd.DataFrame.from_dict(performance)
		performance_df.to_csv(performance_file, index=False)

	def save_features(self, feature_names):
		"""Saves the list of final set of features."""
		features_file = os.path.join(self.output_dir, "features.txt")
		with open(features_file, 'w') as file: 
			for name in feature_names:
				file.write(name + "\n")

	@override 
	def output(self):
		"""Overrides luigi.Task output() to define task outputs."""
		model = os.path.join(self.output_dir, "model.pkl")
		hyperparameters = os.path.join(self.output_dir, "hyperparameters.csv")
		performance = os.path.join(self.output_dir, "cv_performance.csv")
		features = os.path.join(self.output_dir, "features.txt")
		best_performance = os.path.join(self.output_dir, "best_cv_performance.csv")

		output_files = [
			luigi.LocalTarget(model),
			luigi.LocalTarget(hyperparameters),
			luigi.LocalTarget(performance),
			luigi.LocalTarget(features)
		]

		return output_files
