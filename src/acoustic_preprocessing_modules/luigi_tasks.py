import os

import luigi
import pandas as pd 
from typing_extensions import override

from acoustic_preprocessing_modules.base_functions import *
from utility_modules.file import File


class OneToOneTask(luigi.Task):
	parameters = luigi.DictParameter()
	input_file = luigi.Parameter()
	output_file = luigi.Parameter()

	@override
	def requires(self):
		return File(self.input_file)

	@override
	def output(self):
		return luigi.LocalTarget(self.output_file)


class ConvertAudioFileType(OneToOneTask):
	@override
	def run(self):
		input_format = self.parameters['input_format']
		output_format = self.parameters['output_format']
		
		convert_audio_file_type(
			input_file=self.input_file, 
			output_file=self.output_file, 
			input_format=input_format, 
			output_format=output_format
		)

		return None


class ConvertAudioBitDepth(OneToOneTask):
	@override
	def run(self):
		target_bit_depth = self.parameters['target_bit_depth']

		convert_audio_bit_depth(
			input_file=self.input_file, 
			output_file=self.output_file, 
			target_bit_depth=target_bit_depth
		)

		return None


class DenoiseAudioWithLogMMSE(OneToOneTask):
	@override
	def run(self):
		initial_noise = self.parameters['initial_noise']
		window_size = self.parameters['window_size']
		noise_threshold = self.parameters['noise_threshold']

		denoise_audio_with_logmmse(
			input_file=self.input_file, 
			output_file=self.output_file, 
			initial_noise=initial_noise, 
			window_size=window_size, 
			noise_threshold=noise_threshold
		)

		return None


class NormalizeAudioAmplitude(OneToOneTask):
	@override
	def run(self):
		target_dbfs = self.parameters['target_dbfs']

		normalize_audio_amplitude(
			input_file=self.input_file,
			output_file=self.output_file, 
			target_dbfs=target_dbfs
		)

		return None


class ExtractOpenSMILEFeatures(OneToOneTask):
	@override
	def run(self):
		feature_set = self.parameters['feature_set']
		level = self.parameters['level']
		is_for_openxbow = self.parameters['is_for_openxbow']

		extract_opensmile_features(
			input_file=self.input_file,
			output_file=self.output_file, 
			feature_set=feature_set, 
			level=level,
			is_for_openxbow=is_for_openxbow
		)

		return None


class ExtractOpenXBOWFeatures(OneToOneTask):
	@override
	def run(self):
		openxbow_jar_app = self.parameters['openxbow_jar_app']
		audio_book_size = self.parameters['audio_book_size']
		clustering = self.parameters['clustering']
		
		extract_openxbow_features(
			input_file=self.input_file, 
			output_file=self.output_file, 
			openxbow_jar_app=openxbow_jar_app, 
			audio_book_size=audio_book_size, 
			clustering=clustering
		)

		return None
