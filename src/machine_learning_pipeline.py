import argparse
import os 
import shutil


import luigi
import pandas as pd
from typing_extensions import override


from machine_learning.cross_validation import NestedCrossValidationTask
from machine_learning.cross_validation import FlatCrossValidationTask
from utility_modules.config_parser import load_yaml
from utility_modules.file import File


class MachineLearningPipeline(luigi.Task):
	"""
	A luigi.Task class that implements the machine learning training pipeline.
	
	Two tasks are supported:
	1. Flat Cross-Validation
	2. Nested Cross-Validation 
	
	Read more about these protocols: https://doi.org/10.1016/j.eswa.2021.115222
	
	Parameters
	----------
	input_dir : str
		path to the input directory
	output_dir : str
		  path to the output directory
	 config_file : str
		  path to the configuration file

	 Attributes
	 ----------
	_training_protocol : dict
		dictionary of supported training protocols

	 Methods
	 -------
	requires()
		Overrides luigi.Task to check if dependencies exist.
	 get_metadata_config()
	 	Returns the metadata configuration from config_file.
	 get_training_config()
	 	Returns the training configuration from config_file.
	run()
		Overrides luigi.Task run() to run the training pipeline.
	get_training_protocol_task()
		Returns an instance of the training protocol.
	get_pipelines()
		Returns the list of pipeline configurations from config_file.
	copy_config_file()
		Copies configuration file to the output directory.
	generate_summary()
		Generates machine learning training summary.
	output()
		Overrides luigi.Task output() to define the expected training output.
	"""

	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	config_file = luigi.Parameter()

	_training_protocols = {
		'nested_cross_validation': NestedCrossValidationTask,
		'flat_cross_validation': FlatCrossValidationTask
	}

	@override
	def requires(self):
		"""Overrides luigi.Task requires() to check if dependencies exist."""
		config_file = File(self.config_file)
		yield config_file

		metadata_config = self.get_metadata_config()
		metadata_file = os.path.join(
			self.input_dir, 
			metadata_config['file']
		)
		metadata_file = File(metadata_file)
		yield metadata_file

	def get_metadata_config(self):
		"""Returns the metadata configuration from config_file."""
		config = load_yaml(self.config_file)
		return config['metadata_config']

	def get_training_config(self):
		"""Returns the training configuration from config_file."""
		config = load_yaml(self.config_file)
		return config['training_config']

	@override
	def run(self):
		"""Overrides luigi.Task run() to run the training pipeline.
		
		Steps: 
			1. Get training protocol.
			2. For each pipeline in the pipeline configuration:
			2.1. Run training protocol on the pipeline.
			3. Generate a summary in the output directory.
			4. Copy the configuration file to the output directory.
		"""

		training_config = self.get_training_config()
		training_task = self.get_training_protocol_task()
		metadata_config = self.get_metadata_config()
		pipelines = self.get_pipelines()

		for pipeline in pipelines: 
			yield training_task(
				input_dir=self.input_dir, 
				output_dir=self.output_dir,
				metadata_config=metadata_config,
				training_config=training_config,
				pipeline_config=pipeline
			)

		self.copy_config_file()
		self.generate_summary()

	def get_training_protocol_task(self):
		"""Returns an instance of the training protocol."""
		training_config = self.get_training_config()
		
		training_protocol = training_config['training__protocol']
		if training_protocol not in self._training_protocols.keys():
			raise Exception(f"Training protocol not supported: {training_protocol}")
		
		return self._training_protocols[training_protocol]

	def get_pipelines(self):
		"""Returns the list of pipeline configurations from config_file."""
		config = load_yaml(self.config_file)
		return config['pipeline_config']

	def copy_config_file(self):
		"""Copies configuration file to the output directory."""
		filename = os.path.basename(self.config_file)
		dest = os.path.join(self.output_dir, filename)
		shutil.copyfile(self.config_file, dest)

	def generate_summary(self):
		"""Generates machine learning training summary."""
		pipelines = self.get_pipelines()
		summary = {}
		
		for pipeline in pipelines:
			unique_id = pipeline['unique_id']
			performance_file = os.path.join(
				self.output_dir,
				unique_id,
				'performance.csv'
			)
			performance = pd.read_csv(performance_file, index_col=0)
			summary[unique_id + '_mean'] = performance.loc['mean']
			summary[unique_id + '_std'] = performance.loc['std']
		 
		summary_df = pd.DataFrame.from_dict(summary, orient='index')
		summary_file = os.path.join(self.output_dir, 'summary.csv')
		summary_df.to_csv(summary_file, index=True)

	@override
	def output(self):
		"""Overrides luigi.Task output() to define the expected training output."""
		config_file = os.path.basename(self.config_file)
		config_file = os.path.join(self.output_dir, config_file)
		summary_file = os.path.join(self.output_dir, 'summary.csv')
		dependencies = [
			luigi.LocalTarget(config_file),
			luigi.LocalTarget(summary_file)
		]
		return dependencies


def main(input_dir, output_dir, config_file):
	"""Instantiates MachineLearningPipeline and run it with luigi server.
	
	Configure this part as needed: 
	https://luigi.readthedocs.io/en/stable/running_luigi.html

	Parameters
	----------
	input_dir : str
		path to the input directory
	output_dir : str
		path to the output directory
	config_file : str
		path to the configuration file
	"""
	luigi.build([
		MachineLearningPipeline(
			input_dir=input_dir, 
			output_dir=output_dir, 
			config_file=config_file
		)
	])


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-I', '--input', type=str, help='Input Directory Path')
	arg_parser.add_argument('-O', '--output', type=str, help='Output Directory Path')
	arg_parser.add_argument('-C', '--config', type=str, help='Configuration File Path')
	args = arg_parser.parse_args()
	
	input_dir = args.input 
	output_dir = args.output
	config_file = args.config

	main(input_dir, output_dir, config_file)