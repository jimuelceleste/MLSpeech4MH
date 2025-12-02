import os 
import pandas as pd
import argparse

def create_metadata(folder_path):
	"""
	Creates the metadata file inside the provided folder. The subject id is by default the name of the folder and 
	the sample id is by default the name of the file.

	Parameters:
	    folder_path (str): Path to the input file.
	"""
	filenames = []
	sample_ids = []
	subject_ids = []

	subject_id = os.path.basename(folder_path)	# get the name of the folder as subject id
	for filename in os.listdir(folder_path):
		full_path = os.path.join(folder_path, filename)
		if not os.path.isfile(full_path) or filename.lower() == "metadata.csv":
			continue
		sample_id = filename.split('.')[0]
		filenames.append(filename)
		sample_ids.append(sample_id)
		subject_ids.append(subject_id)

	metadata_df = pd.DataFrame()
	metadata_df["filename"] = filenames
	metadata_df["subject_id"] = subject_ids
	metadata_df["sample_id"] = sample_ids
	metadata_df.to_csv(os.path.join(folder_path, "metadata.csv"), index=False)

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-I', '--input', type=str, help='Input Directory Path')
	arg_parser.add_argument('-r', '--recursive', action='store_true')	# a flag that shows whether it is a recursive input folder or not
	args = arg_parser.parse_args()
	
	input_dir = args.input
	is_recursive = args.recursive
	if is_recursive:
		for subject in os.listdir(input_dir):
			subject_input_dir = os.path.join(input_dir, subject)
			if os.path.isdir(subject_input_dir):
				create_metadata(subject_input_dir)
	else:
		create_metadata(input_dir)

	