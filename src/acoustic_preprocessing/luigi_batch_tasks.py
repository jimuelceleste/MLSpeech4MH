import os
import shutil 

import luigi
import pandas as pd 
from typing_extensions import override

from acoustic_preprocessing.luigi_tasks import *
from utility_modules.file import File

class BatchProcessor(luigi.Task):
	"""
	Batch processing of files:
	This task expects a `metadata.csv` file inside `input_dir` (input directory).
	The `metadata.csv` file should contain a column named `filename`.
	This task processes all files defined in the `filename` column of the metadata file.
	All these files should exist inside the `input_dir` directory.
	File processing should be defined under `_process_file()` function (override in subclass).
	This processing should be a `OneToOne` task, i.e., one input and one output (see luigi_tasks.py).

	Parameters:
		parameters (dict): Dictionary of parameters for the task.
		input_dir (str): Path to the input directory.
		output_dir (str): Path to the output directory.
	"""
	parameters = luigi.DictParameter()
	input_dir = luigi.Parameter()
	output_dir = luigi.Parameter()

	@override
	def requires(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		yield File(metadata_file)

	@override
	def run(self):
		os.makedirs(self.output_dir, exist_ok=True)

		for input_file in self._input_iterator():
			yield self._process_file(input_file)
			
		self._update_metadata()

	@override
	def output(self):
		updated_metadata = os.path.join(self.output_dir, "metadata.csv")
		return luigi.LocalTarget(updated_metadata)

	def _input_iterator(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		metadata = pd.read_csv(metadata_file)
		input_files = metadata["filename"]
		for file in input_files:
			input_file = os.path.join(self.input_dir, file)
			yield input_file

	def _output_iterator(self):
		for file in self._input_iterator():
			output_file = self._get_output_file(file)
			yield output_file

	def _get_output_file(self, input_file):
		file = os.path.basename(input_file)
		output_file = os.path.join(self.output_dir, file)
		return output_file

	def _process_file(self, input_file):
		pass

	def _update_metadata(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		metadata = pd.read_csv(metadata_file)
		output_files = []
		for file in self._output_iterator():
			output_file = os.path.basename(file)
			output_files.append(output_file)
		metadata['filename'] = output_files
		metadata.to_csv(self.output().path, index=False)


class BatchConvertAudioFileType(BatchProcessor):
	@override
	def _get_output_file(self, input_file):
		file = os.path.basename(input_file)
		base, ext = os.path.splitext(file)
		output_ext = self.parameters['output_format']
		output_file = os.path.join(self.output_dir, f"{base}.{output_ext}")
		return output_file

	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return ConvertAudioFileType(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file, 
		)


class BatchConvertAudioBitDepth(BatchProcessor):
	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)		
		return ConvertAudioBitDepth(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file, 
		)


class BatchDenoiseAudioWithLogMMSE(BatchProcessor):
	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return DenoiseAudioWithLogMMSE(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file, 
		)


class BatchNormalizeAudioAmplitude(BatchProcessor):
	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return NormalizeAudioAmplitude(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file, 
		)


class BatchExtractOpenSMILEFeatures(BatchProcessor):
	@override
	def _get_output_file(self, input_file):
		file = os.path.basename(input_file)
		base, ext = os.path.splitext(file)
		output_file = os.path.join(self.output_dir, base + ".csv")
		return output_file

	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return ExtractOpenSMILEFeatures(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file,
		)


class BatchExtractOpenXBOWFeatures(BatchProcessor):
	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return ExtractOpenXBOWFeatures(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file, 
		)