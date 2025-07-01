import os 
import pickle 

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import luigi
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from typing_extensions import override

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC
from sklearn.svm import SVR

from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor 

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from machine_learning_modules.metadata import Metadata
from utility_modules.file import File

class HyperparameterOptimizationCVTask(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	training_config = luigi.DictParameter()
	pipeline_config = luigi.DictParameter()
	fold_iterator_file = luigi.Parameter()
	metadata_file = luigi.Parameter()

	optimization_techniques = {
		'grid_search': GridSearchCV,
		'bayes_search': BayesSearchCV
	}

	feature_scaling_tasks = {

	}
	
	feature_selection_tasks = {

	}

	data_augmentation_tasks = {

	}

	classification_models = {

	}

	regression_models = {

	}
	
	machine_learning_tasks = {
		'standard_scaler': StandardScaler,

		'select_k_best': SelectKBest,
		'pca': PCA, 
		'lsa': TruncatedSVD,

		'smote': SMOTE,

		'random_forest_classifier': RandomForestClassifier,
		'support_vector_classifier': SVC, 
		'xgb_classifier': XGBClassifier,
		'logistic_regression_classifier': LogisticRegression,
		'multilayer_perceptron_classifier': MLPClassifier,

		'random_forest_regressor': RandomForestRegressor,
		'support_vector_regressor': SVR,
		'xgb_regressor': XGBRegressor, 
		'linear_regressor': LinearRegression, 
		'multilayer_perceptron_regressor': MLPRegressor
	}

	@override
	def requires(self):
		# Metadata
		yield File(self.metadata_file)

		# Fold iterator 
		yield File(self.fold_iterator_file)

		# Features
		for file in self._input_iterator():
			yield File(file)
	
	@override 
	def run(self):
		optimizer = self.get_optimizer()
		metadata = self.load_metadata()
		train_features = self.load_features(metadata)
		train_labels = self.load_labels(metadata)

		optimizer.fit(train_features, train_labels)
		model = optimizer.best_estimator_.fit(train_features, train_labels)
		hyperparameters = optimizer.best_params_
		performance = optimizer.cv_results_
		best_index = optimizer.best_index_
		features = model[:-1].get_feature_names_out()

		os.makedirs(self.output_dir, exist_ok=True)
		self.save_model(model)
		self.save_hyperparameters(hyperparameters)
		self.save_performance(performance)
		self.save_features(features)
		self.save_best_performance(performance, best_index)

	@override 
	def output(self):
		model = os.path.join(self.output_dir, "model.pkl")
		hyperparameters = os.path.join(self.output_dir, "hyperparameters.csv")
		performance = os.path.join(self.output_dir, "performance.csv")
		best_performance = os.path.join(self.output_dir, "best_performance.csv")
		features = os.path.join(self.output_dir, "features.txt")
		output_files = [
			luigi.LocalTarget(model),
			luigi.LocalTarget(hyperparameters),
			luigi.LocalTarget(performance),
			luigi.LocalTarget(features)
		]
		return output_files

	def load_metadata(self):	
		label_column = self.training_config['metadata__label_column']
		metadata = Metadata(self.metadata_file, label_column)		
		return metadata

	def _input_iterator(self):
		metadata = self.load_metadata()
		input_files = metadata.get_filenames()
		for file in input_files:
			input_file = os.path.join(self.input_dir, file)
			yield input_file

	def load_features(self, metadata):
		filenames = metadata.get_filenames()
		features_list = []
		for file in filenames:
			features_file = os.path.join(self.input_dir, file)
			features = pd.read_csv(features_file)
			features_list.append(features)	
		features_list = pd.concat(features_list)
		return features_list

	def load_labels(self, metadata):
		labels = metadata.get_labels()
		return labels.values.tolist()

	def build_pipeline(self):
		steps = self.pipeline_config['steps']
		pipeline = []
		for step in steps.keys():
			task = steps[step]
			if task not in self.machine_learning_tasks.keys():
				raise Exception(f'Machine learning task not supported: {task}')
			
			# MODIFY: Make sure to edit this to capture the general case! 
			# THAT IS: Ensure that parameters are all declared in the config file.
			if (task == 'select_k_best') and (self.training_config['task'] == 'regression'):
				task_instance = self.machine_learning_tasks[task](score_func=f_regression)
			else: 
				task_instance = self.machine_learning_tasks[task]()

			pipeline.append((step, task_instance))
		return Pipeline(pipeline)

	def build_parameter_grid(self):
		parameters = dict(self.pipeline_config['parameters'])
		# MODIFY: Convert function parameters as functions.
		return parameters
	
	def load_pickle(self, file):
		with open(file, 'rb') as file:
			return pickle.load(file)

	def load_fold_iterator(self):
		iterator = self.load_pickle(self.fold_iterator_file)
		return iterator

	def get_optimization_scoring(self):
		return self.training_config['optimization__scoring']

	def get_optimization_task(self):
		technique = self.training_config['optimization__technique']
		if technique not in self.optimization_tasks.keys():
			raise Exception('Optimization technique not supported.')
		optimizer = self.optimization_tasks[technique]
		return optimizer

	def get_optimizer(self):
		optimization_technique = self.training_config['optimization__technique']
		if optimization_technique not in self.optimization_techniques.keys():
			raise Exception('Optimization technique not supported.')
		
		pipeline = self.build_pipeline()
		parameter_grid = self.build_parameter_grid()
		fold_iterator = self.load_fold_iterator()
		optimization_scoring = self.get_optimization_scoring()

		optimizer = self.optimization_techniques[optimization_technique](
			estimator=pipeline,
			param_grid=parameter_grid,
			cv=fold_iterator,
			scoring=optimization_scoring,
			n_jobs=-1, 
			return_train_score=True,
			verbose=True
		)

		return optimizer
		
	def save_model(self, model):
		model_file = os.path.join(self.output_dir, "model.pkl")
		with open(model_file, 'wb') as file:
			pickle.dump(model, file)

	def save_hyperparameters(self, hyperparameters):
		hyp_file = os.path.join(self.output_dir, "hyperparameters.csv")

		for key in hyperparameters.keys():
			val = hyperparameters[key]
			if isinstance(val, tuple) or isinstance(val, list):
				hyperparameters[key] = " ".join(str(x) for x in val)

		hyp_df = pd.DataFrame(hyperparameters, index=[0])
		hyp_df.to_csv(hyp_file, index=False)

	def save_performance(self, performance):
		performance_file = os.path.join(self.output_dir, "performance.csv")
		performance_df = pd.DataFrame.from_dict(performance)
		performance_df.to_csv(performance_file, index=False)

	def save_features(self, feature_names):
		features_file = os.path.join(self.output_dir, "features.txt")
		with open(features_file, 'w') as file: 
			for name in feature_names:
				file.write(name + "\n")

	def save_best_performance(self, performance, best_index):
		performance_file = os.path.join(self.output_dir, "best_performance.csv")
		performance_df = pd.DataFrame.from_dict(performance)
		best_performance = performance_df.iloc[best_index]
		best_performance.to_csv(performance_file, index=True)