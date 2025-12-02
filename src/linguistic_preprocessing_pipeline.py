import argparse
import os 
import shutil

import luigi
import pandas as pd

from linguistic_preprocessing.luigi_batch_tasks import *
from utility_modules.config_parser import parse_pipeline_config
from utility_modules.file import File


class LinguisticPreprocessingPipeline(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	config_file = luigi.Parameter()
	acoustic_preprocessing_tasks = {
		'extract_lftk_features': BatchExtractLFTKFeatures,
	}

	@override
	def requires(self):
		# Configuration manifest
		yield File(self.config_file)

	@override
	def run(self):
		tasks, _, _ = parse_pipeline_config(self.config_file)
		for task in tasks:
			# Task properties
			task = tasks[task]
			task_name = task['name']
			task_id = task['unique_id']
			task_dependency = task['dependency']
			task_parameters = task['parameters']

			# Check-point: Is task supported?
			if task_name not in self.acoustic_preprocessing_tasks.keys():
				raise Exception(f"Task `{task_name}` not supported. Please review the list of tasks.")

			# Input directory
			if task_dependency == 'input':
				input_dir = self.input_dir
			else:
				input_dir = os.path.join(self.output_dir, task_dependency)
			
			# Output directory
			output_dir = os.path.join(self.output_dir, task['unique_id'])
			
			# Running batch task
			batch_task = self.acoustic_preprocessing_tasks[task_name]
			yield batch_task(
				parameters=task_parameters,
				input_dir=input_dir, 
				output_dir=output_dir
			)

		# Output file
		shutil.copyfile(self.config_file, self.output().path)

	@override
	def output(self):
		basename = os.path.basename(self.config_file)
		output_file = os.path.join(self.output_dir, basename)
		return luigi.LocalTarget(output_file)


def main(input_dir, output_dir, config_file):
	luigi.build(
		[LinguisticPreprocessingPipeline(
			input_dir=input_dir, 
			output_dir=output_dir, 
			config_file=config_file
		)], 
		# local_scheduler=True
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

