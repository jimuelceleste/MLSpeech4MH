import os
import shutil 

import lftk 
import luigi
import pandas as pd
import spacy 
from typing_extensions import override

from utility_modules.file import File
from linguistic_preprocessing.luigi_tasks import *


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


class BatchExtractLFTKFeatures(BatchProcessor):
	@override
	def _get_output_file(self, input_file):
		file = os.path.basename(input_file)
		base, ext = os.path.splitext(file)
		output_file = os.path.join(self.output_dir, base + ".csv")
		return output_file

	@override
	def _process_file(self, input_file):
		output_file = self._get_output_file(input_file)
		return ExtractLFTKFeatures(
			parameters=self.parameters,
			input_file=input_file, 
			output_file=output_file,
		)


class BatchExtractLFTKFeatures_Archive(luigi.Task):
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

		for input_file in self._input_iterator():
			yield File(input_file)

	@override
	def run(self):
		os.makedirs(self.output_dir, exist_ok=True)

		files = [file for file in self._input_iterator()]
		docs = self.get_docs(files)
		extractor = lftk.Extractor(docs=docs)
		features_list = lftk.search_features(
			domain=self.parameters['domain'],
			family=self.parameters['family'],
			language=self.parameters['language'],
			return_format='list_key'
		)
		features = extractor.extract(features=features_list)

		for i, (current_features) in enumerate(features):
			current_features_df = pd.DataFrame.from_dict([current_features])
			filename = files[i]
			current_output_file = self._get_output_file(filename)
			current_features_df.to_csv(current_output_file, index=False)

		self._update_metadata()

	@override
	def output(self):
		updated_metadata = os.path.join(self.output_dir, "metadata.csv")
		output_files = [luigi.LocalTarget(updated_metadata)]

		for file in self._input_iterator():
			output_file = self._get_output_file(file)
			output_file = luigi.LocalTarget(output_file)
			output_files.append(output_file)

		return output_files

	def get_docs(self, files):
		nlp = spacy.load("en_core_web_sm")
		docs = []
		for file in files:
			with open(file, "r") as f:
				text = f.read()
				text = text.replace('\n', ' ') # replace new line with spaces
			doc = nlp(text)
			docs.append(doc)
		return docs 

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

	def _update_metadata(self):
		metadata_file = os.path.join(self.input_dir, "metadata.csv")
		metadata = pd.read_csv(metadata_file)
		output_files = []
		for file in self._output_iterator():
			output_file = os.path.basename(file)
			output_files.append(output_file)
		metadata['filename'] = output_files
		new_metadata = os.path.join(self.output_dir, "metadata.csv")
		metadata.to_csv(new_metadata, index=False)

	def _get_output_file(self, input_file):
		file = os.path.basename(input_file)
		base, ext = os.path.splitext(file)
		output_file = os.path.join(self.output_dir, base + ".csv")
		return output_file
