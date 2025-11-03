import os
import subprocess 

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf

from logmmse import logmmse_from_file
from pydub import AudioSegment 


def convert_audio_file_type(input_file, output_file, input_format, output_format):
	"""
	Converts audio file from one type to another using pydub.AudioSegment.
	See documentation: https://github.com/jiaaro/pydub

	Parameters:
	    input_file (str): Path to the input file.
	    output_file (str): Path for the output file.
	    input_format (str): File type of the input file.
	    output_format (str): File type for the output file.
	"""

	audio = AudioSegment.from_file(
		input_file, 
		format=input_format
	)

	audio_buffer = audio.export(
		output_file, 
		format=output_format
	)

	return None 


def convert_audio_bit_depth(input_file, output_file, target_bit_depth):
	"""
	Converts bit depth of an audio file with soundfile.
	This conversion is necessary for audio denoising with logmmse.
	See denoise_with_logmmse function.
	Supported bit-depth formats:
	1. 'PCM_24' for signed 24 bit PCM,
 	2. 'PCM_16' for signed 16 bit PCM, and
 	3. 'PCM_S8' for signed 8 bit PCM'.
 	See https://python-soundfile.readthedocs.io/en/0.13.1/.

	Parameters:
		input_file (str): Path to the input file.
		output_file (str): Path to the output file.
		target_bit_depth (str): The new bit-depth.

	"""
	data, samplerate = sf.read(input_file)
	
	sf.write(
		output_file, 
		data, 
		samplerate=samplerate, 
		subtype=target_bit_depth
	)

	return None


def denoise_audio_with_logmmse(input_file, output_file, initial_noise=6, window_size=0, noise_threshold=0.15):
	"""
	Denoises audio file with LogMMSE.
	Input file should have the following format:
	1. wav file; 
	2. bit-depth: 
		a. 32-bit floating-point, 
		b. 32-bit PCM, 
		c. 16-bit PCM, or
		d. 8-bit PCM.

	Parameter: 
		input_file (str): Path to the input file.
		output_file (str): Path for the output file. 
		initial_noise (float): 
		window_size (float):	
		noise_threshold (float):
	"""
	logmmse_from_file(
		input_file,
		output_file,
		initial_noise,
		window_size,
		noise_threshold
	)

	return None


def normalize_audio_amplitude(input_file, output_file, target_dbfs):
	"""
	Normalize audio amplitude with pydub.AudioSegment.
	
	Parameters:
		input_file (str): Path to the input file.
		output_file (str): Path for the output file.
		target_dbfs (int): Target DBFS.  
	
	"""
	audio = AudioSegment.from_wav(input_file)
	
	change_in_dbfs = target_dbfs - audio.dBFS 
	audio = audio.apply_gain(change_in_dbfs)
	
	audio.export(
		output_file, 
		format='wav', 
		parameters=['-ar', '16000', '-ac', '1']
	)

	return None


def extract_opensmile_features(input_file, output_file, feature_set, level, is_for_openxbow):
	"""
	Extracts features from an audio file with openSMILE.
	See openSMILE documentation: https://audeering.github.io/opensmile-python/

	Parameters:
		input_file (str): Path to the input file. 
		output_file (str): Path for the output file. 
		feature_set (str): One of the three feature sets: `compare_2016`, `egemaps`, `gemaps`
		level (str): One of the three levels: `functionals`, `lld`, `lld_deltas`
		is_for_openxbow (bool): True if the feature will be used for openXBOW feature extraction.
	"""
	feature_sets = {
		'compare_2016': opensmile.FeatureSet.ComParE_2016,  
		'egemaps': opensmile.FeatureSet.eGeMAPSv02, 
		'gemaps': opensmile.FeatureSet.GeMAPSv01b 
	}

	levels = {
		'functionals': opensmile.FeatureLevel.Functionals, 
		'lld': opensmile.FeatureLevel.LowLevelDescriptors, 
		'lld_deltas': opensmile.FeatureLevel.LowLevelDescriptors_Deltas
	}

	feature_set = feature_sets[feature_set] 
	level = levels[level]
	
	smile = opensmile.Smile(
		feature_set=feature_set, 
		feature_level=level
	)
	
	features = smile.process_file(input_file)
	features = features.reset_index()
	features = features.drop(['start', 'end'], axis='columns')
	
	if not is_for_openxbow:
		features = features.drop(['file'], axis='columns')

	features.to_csv(output_file, sep=',', index=False)

	return None 
	

def extract_openxbow_features(input_file, output_file, openxbow_jar_app, audio_book_size, clustering):
	"""
	Uses openXBOW to extract Bag-of-Audio-Word features from LLD features (see openSMILE features).
	This function expects that openXBOW jar file and java interpreter are installed.
	See openXBOW documentation: https://github.com/openXBOW/openXBOW

	Parameters:
		input_file (str): Path to the input file.
		output_file (str): Path for the output file.
		openxbow_jar_app (str): Path to the openXBOW jar app. 
		audio_book_size (int): Number of patterns to look for from the input data (i.e., LLD).
		clustering (str): Either `random` (random sampling), `kmeans++` (kmeans++), or `kmeans` (standard kmeans)
	"""

	filename, ext = os.path.splitext(output_file)
	temp_file = filename + "_temp" + ext

	command = [
		'java', 
		'-jar', openxbow_jar_app, 
		'-i', input_file, 
		'-o', temp_file, 
		'-size', str(audio_book_size),
		'-c', clustering
	]

	subprocess.run(command)

	# Post-processing
	features = np.loadtxt(temp_file, delimiter=';')
	columns = np.arange(start=1, stop=len(features) + 1, step=1)
	formatted_features = pd.DataFrame(data=[features], columns=columns)
	formatted_features.to_csv(output_file, index=False)
	os.remove(temp_file)

	return None

def extract_deepspectrum_features(input_file, output_file, threads_number, batch_size, extraction_network, feature_layer):
	"""
	Extracts DeepSpectrum features from an audio file.
	See DeepSpectrum documentation: https://github.com/DeepSpectrum/DeepSpectrum
	Parameters:
		input_file (str): Path to the input file.
		output_file (str): Path for the output file.
		threads_number (int): Number of threads to be used.
		batch_size (int): Batch size for processing multiple audio files.
		extraction_network (str): The CNN model architecture to be used as the feature extractor (i.e., encoder) for converting audio spectrograms into embeddings.
		feature_layer (str): Name of the layer from which features should be extracted.
	"""

	command = [
		"python", "-m", 
		'deepspectrum',
		'features', 
		input_file,
		'-t', str(threads_number), str(batch_size),
		'-nl',	# remove labels from the output
		'-en', extraction_network,
		'-fl', feature_layer,
		'-o', output_file
	]

	subprocess.call(command)

	return None