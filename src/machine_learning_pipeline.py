import argparse
import os 
import shutil

import luigi
import pandas as pd
from typing_extensions import override

from machine_learning_modules.fold_generation_tasks import SubjectWiseFoldGenerationTask
from machine_learning_modules.cross_validation_tasks import NestedCrossValidationTask
from machine_learning_modules.cross_validation_tasks import FlatCrossValidationTask
from utility_modules.config_parser import load_yaml
from utility_modules.file import File


class MachineLearningPipeline(luigi.Task):
	"""
	Trains and validates machine learning models with specified protocol.
	Protocols include: nested and flat cross-validation.
	Read more about these protocols: https://doi.org/10.1016/j.eswa.2021.115222
	A `metadata.csv` file is expected in the input directory.
	This file should contain the following columns:
	1. `filename` - filenames of the feature files 
	2. `record_id` - unique id for the file 
	3. `subject_id` - unique id for the participant/patient/person
	4. `label` - the predicted label, this could be named with anything
	The label could be named anything as this is specified in the configuration file.
	The feature files are expected to be csv files.

	Parameters: 
		input_dir (str): Path to the input directory.
		output_dir (str): Path to the output directory.
		config_file (str): Path to the configuration manifest file.
	"""
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	config_file = luigi.Parameter()

	training_techniques = {
		'nested_cross_validation': NestedCrossValidationTask,
		'flat_cross_validation': FlatCrossValidationTask
	}

	@override
	def requires(self):
		# Require configuration manifest
		config_file = File(self.config_file)
		yield config_file

		# Require metadata file
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		metadata_file = File(metadata_file)
		yield metadata_file

	@override
	def run(self):
		training_config, pipelines = self.parse_config()

		# Check-point: Is the training technique supported? 
		training_technique = training_config['technique']
		
		if training_technique not in self.training_techniques.keys():
			raise Exception("Training technique not supported.")

		# Run training technique for all tasks 
		training_task = self.training_techniques[training_technique]
		training_parameters = training_config['parameters']
		
		for pipeline in pipelines: 
			yield training_task(
				input_dir=self.input_dir, 
				output_dir=self.output_dir,
				training_config=training_parameters,
				pipeline_config=pipeline
			)

		# Copy configuration file to the output directory
		shutil.copyfile(self.config_file, self.output().path)

	@override
	def output(self):
		config_file = os.path.basename(self.config_file)
		config_file = os.path.join(self.output_dir, config_file)
		return luigi.LocalTarget(config_file)

	def get_pipelines(self):
		config = load_yaml(self.config_file)
		pipelines = config['pipeline_config']
		return pipelines

	def get_training_config(self):
		config = load_yaml(self.config_file)
		training_config = config['training_config']
		return training_config

	def parse_config(self):
		training_config = self.get_training_config()
		pipelines = self.get_pipelines()
		return training_config, pipelines


def main(input_dir, output_dir, config_file):
	luigi.build(
		[MachineLearningPipeline(
			input_dir=input_dir, 
			output_dir=output_dir, 
			config_file=config_file
		)]
	)


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