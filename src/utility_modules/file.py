import luigi 
from typing_extensions import override

class File(luigi.ExternalTask):
	"""
	Luigi task representing a local file via luigi.LocalTarget() function.
	
	Attributes:
		file (str): Path to the file.
	"""
	file = luigi.Parameter()

	@override
	def output(self):
		return luigi.LocalTarget(self.file)