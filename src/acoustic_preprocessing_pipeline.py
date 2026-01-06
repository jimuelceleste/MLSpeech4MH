import argparse
import os 
import shutil

import luigi
try:
    import importlib.metadata  # stdlib name on py>=3.8
except ImportError:
    import importlib_metadata as _im
    import sys, types
    # create a fake package "importlib" if needed
    if "importlib" not in sys.modules:
        importlib_pkg = types.ModuleType("importlib")
        sys.modules["importlib"] = importlib_pkg
    # expose importlib.metadata -> backport
    sys.modules["importlib.metadata"] = _im
	
from acoustic_preprocessing.luigi_batch_tasks import *
from acoustic_preprocessing.luigi_tasks import *
from utility_modules.config_parser import parse_pipeline_config
from utility_modules.file import File
from create_metadata_files import create_metadata


class AcousticPreprocessingPipeline(luigi.Task):
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()
	config_file = luigi.Parameter()
	acoustic_preprocessing_tasks = {
		'convert_audio_file_type': BatchConvertAudioFileType,
		'convert_audio_bit_depth': BatchConvertAudioBitDepth,
		'denoise_audio_with_logmmse': BatchDenoiseAudioWithLogMMSE,
		'normalize_audio_amplitude': BatchNormalizeAudioAmplitude,
		'extract_opensmile_features': BatchExtractOpenSMILEFeatures,
		'extract_openxbow_features': BatchExtractOpenXBOWFeatures,
		'extract_deepspectrum_features': BatchExtractDeepSpectrumFeatures,
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
		[AcousticPreprocessingPipeline(
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
	arg_parser.add_argument('-r', '--recursive', action='store_true')	# a flag that shows whether it is a recursive input folder or not
	args = arg_parser.parse_args()
	
	input_dir = args.input 
	output_dir = args.output
	config_file = args.config
	is_recursive = args.recursive
	
	# python src/acoustic_preprocessing_pipeline.py -I "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/data/TAUKADIAL2024" -O "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/results/TAUKADIAL2024/acoustic" -C "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/config/TAUKADIAL2024_acoustic.yml"	
	# input_dir = "D:\\Study\\Projects\\eMPowerProject\\results"
	# output_dir = "D:\\Study\\Projects\\eMPowerProject\\acoustic_results_deepspectrum"
	# config_file = "D:\\Study\\Projects\\MLSpeech4MH\\config\\eMPower_acoustic.yml"
	# is_recursive = True

	if is_recursive:
		for subject in os.listdir(input_dir):
			subject_input_dir = os.path.join(input_dir, subject)
			if os.path.isdir(subject_input_dir):
				create_metadata(subject_input_dir)
				subject_output_dir = os.path.join(output_dir, subject)
				main(subject_input_dir, subject_output_dir, config_file)
	else:
		create_metadata(input_dir)
		main(input_dir, output_dir, config_file)