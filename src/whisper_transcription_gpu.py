import argparse
import json
import os 
import time 

import whisper 


def transcribe(input_file, model):
    return model.transcribe(input_file)


def save_result(transcript, output_file):
    with open(output_file, 'w') as f:
        json.dump(transcript, f, indent=4)
    return None 


def batch_transcribe(input_dir, output_dir, formats, model):
    for root, directories, files in os.walk(input_dir):
        n = len(files)
        i = 1
        
        for file in files:
            print(f"{i}/{n}: {file}")
            filename, ext = os.path.splitext(file)
            if ext.lstrip('.') not in formats:
                print("Skipping file: file does not have the set format.")
                i += 1
                continue

            input_file = os.path.join(root, file)
            print(f"input file: {input_file}")
            output_file = os.path.join(output_dir, filename + ".json")
            print(f"output_file: {output_file}")

            if os.path.isfile(output_file):
                print("Skipping transcription: file is previously processed!")
                i += 1
                continue

            result = transcribe(input_file, model)
            save_result(result, output_file)
            print("Transcription saved")
            i += 1

    return None 


def main(input_dir, output_dir, formats):
    start_time = time.time()
    
    model = whisper.load_model("turbo", device="cuda")
    batch_transcribe(input_dir, output_dir, formats, model)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Time summary")
    print(f"start: {start_time}")
    print(f"end: {end_time}")
    print(f"elapsed time: {elapsed_time}")
    
    output_file = os.path.join(output_dir, "time.txt")
    with open(output_file, "w") as f:
        f.write(f"start: {start_time}")
        f.write(f"end: {end_time}")
        f.write(f"elapsed time: {elapsed_time}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-I", "--input", type=str, help="Input Directory Path")
    arg_parser.add_argument("-O", "--output", type=str, help="Output Directory Path")
    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    formats = ["wav"] # change as needed

    main(input_dir, output_dir, formats)