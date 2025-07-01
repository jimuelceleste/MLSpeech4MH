import argparse
import os 
import shutil 

import luigi
import pandas as pd
from typing_extensions import override

from utility_modules.file import File 
from machine_learning_modules.metadata import Metadata

class BatchMergeFeatures(luigi.Task):
	"""
	Merges features by concatenation.
	See a good reference: 
	https://www.geeksforgeeks.org/early-fusion-vs-late-fusion-in-multimodal-data-processing/
	This task expects a `metadata.csv` file with `filename` column containing the list of files. 
	The files are expected to contain "clean" features:
	1. First row contains the column names 
	2. Second row contains the features
	Note that this task only supports 1xD dimensional feature concatenation (1 row, D features).

	Parameters:
		input_dirs (list): List of directories containing the features.
		output_dir (str): Path to the output directory.
	"""
	input_dirs = luigi.ListParameter()
	output_dir = luigi.Parameter()

	@override
	def requires(self):
		for input_dir in self.input_dirs:
			metadata = os.path.join(input_dir, "metadata.csv")
			yield File(metadata)

			for input_file in self._input_iterator(input_dir):
				yield File(input_file)

	@override 
	def run(self):
		os.makedirs(self.output_dir, exist_ok=True)

		# Check-point: Compare metadata.csv files to confirm similar filenames 
		# The first metadata file will be used as the reference list.
		reference_file = ""
		reference_list = []
		for directory in self.input_dirs:
			metadata_file = os.path.join(directory, "metadata.csv")
			metadata = Metadata(metadata_file, label_column="")
			current_list = metadata.get_filenames()

			if not reference_file:
				reference_file = metadata_file 
				reference_list = current_list
				continue

			if set(current_list) != set(reference_list):
				raise Exception("Features being merged are not compatible.")

		# Merge features 
		for file in reference_list:
			yield MergeFeatures(
				input_file=file,
				input_dirs=self.input_dirs, 
				output_dir=self.output_dir
			)

		# Generate metadata
		output_metadata = os.path.join(self.output_dir, "metadata.csv")
		shutil.copy(reference_file, output_metadata)

	@override 
	def output(self):
		metadata_file = os.path.join(self.output_dir, "metadata.csv")
		output_files = [luigi.LocalTarget(metadata_file)]

		reference_metadata = os.path.join(self.input_dirs[0], "metadata.csv")
		metadata = Metadata(reference_metadata, label_column="")
		for file in metadata.get_filenames():
			output_file = os.path.join(self.output_dir, file)
			output_files.append(luigi.LocalTarget(output_file))
		
		return output_files

	def _input_iterator(self, input_dir):
		for input_dir in self.input_dirs: 
			metadata_file = os.path.join(input_dir, "metadata.csv")
			metadata = Metadata(metadata_file, label_column="")

			for file in metadata.get_filenames():
				input_file = os.path.join(input_dir, file)
				yield input_file


class MergeFeatures(luigi.Task):
	input_file = luigi.Parameter() 
	input_dirs = luigi.ListParameter()
	output_dir = luigi.Parameter()

	@override
	def requires(self):
		for input_file in self._input_iterator():
			yield File(input_file)
	
	@override 
	def run(self):
		input_files = [file for file in self._input_iterator()]
		output_file = os.path.join(self.output_dir, self.input_file)
		merge_features(input_files, output_file)

	@override
	def output(self):
		output_file = os.path.join(self.output_dir, self.input_file)
		return luigi.LocalTarget(output_file)

	def _input_iterator(self):
		for input_dir in self.input_dirs:
			input_file = os.path.join(input_dir, self.input_file)
			yield input_file


def merge_features(input_files, output_file):
	"""
	Concatenates features
	Assumes that there's only one row of features

	Parameters:
		input_files (list): List of the input files path.
	"""
	features_list = []
	for file in input_files:
		features = pd.read_csv(file)
		features_list.append(features)
	
	concatenated_features = pd.concat(features_list, axis='columns')
	concatenated_features.to_csv(output_file, index=False)

	return None


def main(input_dirs, output_dir):
	luigi.build(
		[BatchMergeFeatures(
			input_dirs=input_dirs, 
			output_dir=output_dir
		)]
	)


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-I','--input_dirs', nargs='+', help='<Required> Input Directories', required=True)
	arg_parser.add_argument('-O', '--output_dir', type=str, help='Output Directory Path', required=True)
	args = arg_parser.parse_args()

	input_dirs = args.input_dirs
	output_dir = args.output_dir
	# input_dirs = [
		# "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/results/ADrESS2020/acoustic/egemaps_functionals",
		# "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/results/ADrESS2020/linguistic/lftk_features"
	# ]
	# output_dir = "/Users/jimuelcelestejr/Documents/codebook/MLSpeech4MH/results/ADrESS2020/merged_features/egemaps_lftk_features"

	main(input_dirs, output_dir)

