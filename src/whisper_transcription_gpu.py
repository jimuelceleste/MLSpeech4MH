import argparse
import json
import os 
import time
import pandas as pd 
import torch
import whisper 


def transcribe(input_file, model):
    return model.transcribe(input_file)


def save_result(transcript, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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

def transfer_to_csv(output_dir):
    # Get list of JSON files in the output directory
    if not os.path.isdir(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return

    json_files = [file for file in os.listdir(output_dir) if file.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {output_dir}")

    if not json_files:
        return

    for json_file in json_files:
        src_path = os.path.join(output_dir, json_file)
        print(f"Processing JSON: {src_path}")

        with open(src_path, 'r') as f:
            array = json.load(f)

        full_transcript = ""
        segmented_transcript = []
        segmented_transcript_without_short_segment = []

        # extract full transcript
        if "text" in array:
            full_transcript = array["text"]

        # extract segments
        if "segments" in array:
            for item in array["segments"]:
                transcript = item["text"]
                start_time = item["start"]
                end_time = item["end"]
                if transcript != "":
                    segmented_transcript.append([start_time, end_time, transcript])
                    if len(transcript.rstrip()) > 2:
                        segmented_transcript_without_short_segment.append(
                            [start_time, end_time, transcript]
                        )

        # Add another column for pause length before each line
        for i in range(len(segmented_transcript)):
            if i == 0:
                segmented_transcript[i].append('')
            else:
                pause_length = segmented_transcript[i][0] - segmented_transcript[i-1][1]
                segmented_transcript[i].append(pause_length)

        for i in range(len(segmented_transcript_without_short_segment)):
            if i == 0:
                segmented_transcript_without_short_segment[i].append('')
            else:
                pause_length = (
                    segmented_transcript_without_short_segment[i][0]
                    - segmented_transcript_without_short_segment[i-1][1]
                )
                segmented_transcript_without_short_segment[i].append(pause_length)

        # # Join segments based on pause
        # segmented_transcripts_with_joined_segments = segmented_transcript_without_short_segment
        # i = 0
        # while i < len(segmented_transcripts_with_joined_segments) - 1:
        #     if segmented_transcripts_with_joined_segments[i+1][3] < 0.6:
        #         segmented_transcripts_with_joined_segments[i][2] += segmented_transcripts_with_joined_segments[i+1][2]
        #         segmented_transcripts_with_joined_segments[i][1] = segmented_transcripts_with_joined_segments[i+1][1]
        #         del segmented_transcripts_with_joined_segments[i+1]
        #     else:
        #         i += 1

        # Write csv file for full transcript
        dest_file_full_transcript = json_file.split('.json')[0] + '_full_transcript.csv'
        with open(os.path.join(output_dir, dest_file_full_transcript), "w+", encoding="utf-8") as csv_file:
            csv_file.write(full_transcript)

        # Write csv file for segmented transcript
        segmented_transcripts_df = pd.DataFrame(
            segmented_transcript,
            columns=['start_time', 'end_time', 'transcript', 'pause_length_before']
        )
        dest_file_segmented_transcript = json_file.split('.json')[0] + '_segmented_transcript.csv'
        segmented_transcripts_df.to_csv(
            os.path.join(output_dir, dest_file_segmented_transcript),
            index=False,
            encoding="utf-8"
        )

        # Write csv file for segmented transcript with the short text removed
        segmented_transcripts_without_short_segment_df = pd.DataFrame(
            segmented_transcript_without_short_segment,
            columns=['start_time', 'end_time', 'transcript', 'pause_length_before']
        )
        dest_file_segmented_transcript_without_short_segment = (
            json_file.split('.json')[0] + '_segmented_transcript_without_short_segment.csv'
        )
        segmented_transcripts_without_short_segment_df.to_csv(
            os.path.join(output_dir, dest_file_segmented_transcript_without_short_segment),
            index=False,
            encoding="utf-8"
        )

        print(f"CSV files written for {json_file}")



def main(input_dir, output_dir, formats):
    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model on {device}...")
    model = whisper.load_model("turbo", device=device)

    batch_transcribe(input_dir, output_dir, formats, model)
    transfer_to_csv(output_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Time summary")
    print(f"start: {start_time}")
    print(f"end: {end_time}")
    print(f"elapsed time: {elapsed_time}")
    
    # output_file = os.path.join(output_dir, "time.txt")
    # with open(output_file, "w") as f:
    #     f.write(f"start: {start_time}")
    #     f.write(f"end: {end_time}")
    #     f.write(f"elapsed time: {elapsed_time}")



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-I", "--input", type=str, help="Input Directory Path")
    arg_parser.add_argument("-O", "--output", type=str, help="Output Directory Path")
    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    formats = ["wav"] # change as needed

    input_dir = "D:\\Study\\eMPowerProject\\check_in_recordings_wav_cleaned\\denoised_and_normalized\\PAR1"
    output_dir = "D:\\Study\\eMPowerProject\\results\\test"

    main(input_dir, output_dir, formats)