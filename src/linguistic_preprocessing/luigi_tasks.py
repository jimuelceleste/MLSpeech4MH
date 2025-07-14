import luigi
from typing_extensions import override

from linguistic_preprocessing.base_functions import *
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


class ExtractLFTKFeatures(OneToOneTask):
	@override
	def run(self):
		domain = self.parameters['domain']
		family = self.parameters['family']
		language = self.parameters['language']
		
		extract_lftk_features(
			input_file=self.input_file, 
			output_file=self.output_file, 
			domain=domain,
			family=family,
			language=language
		)

		return None