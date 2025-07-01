import lftk 
import spacy
import pandas as pd 

def extract_lftk_features(input_file, output_file, domain, family, language):
	nlp = spacy.load("en_core_web_sm")

	features_list = lftk.search_features(
	    domain=domain,
		family=family,
		language=language,
		return_format='list_key'
	)

	with open(input_file, "r") as f:
		# Prepare text
		text = f.read()
		doc = nlp(text)
		
		# Extract features
		extractor = lftk.Extractor(docs=doc)
		features = extractor.extract(features=features_list)
		
		# Save features
		features_df = pd.DataFrame.from_dict([features])
		features_df.to_csv(output_file, index=False)