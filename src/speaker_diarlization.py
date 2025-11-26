import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import os
import argparse


def pyannote_speaker_diarlization(audio_file, output_file):
    # Community-1 open-source speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token="{huggingface-token}")

    # send pipeline to GPU (when available)
    # pipeline.to(torch.device("cuda"))

    # apply pretrained pipeline (with optional progress hook)
    with ProgressHook() as hook:
        output = pipeline(audio_file, hook=hook, num_speakers=2)  # runs locally

    # save the result
    segments = []
    for turn, speaker in output.speaker_diarization:
        segments.append({
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "speaker": speaker
        })   

    # merge consecutive segments with the same speaker
    merged_segments = []
    current = segments[0]
    for segment in segments[1:]:
        if segment["speaker"] == current["speaker"]:
            current["end_time"] = segment["end_time"]
        else:
            merged_segments.append(current)
            current = segment
    merged_segments.append(current)

    # save the merged segments to csv file
    df = pd.DataFrame(merged_segments)
    df.to_csv(output_file, index=False)

def audio_segment(audio_file, transcript_path):
    transcript = pd.read_csv(transcript_path, index_col=False)
    for index, row in transcript.iterrows():
        start = float(row['start_time']) * 1000
        stop = float(row['end_time']) * 1000
        # print(str(start)+"-"+str(stop))
        new_audio_file = audio_file.split('_')[0] + '_' + str(index + 1).zfill(3) + '.wav'
        audio = AudioSegment.from_wav(f"./{audio_file}")
        audio_segment = audio[start:stop]
        destination = Path(f"./{audio_file.split('_')[0]}_segments")
        Path.mkdir(destination, parents=True, exist_ok=True)
        audio_segment.export(os.path.join(destination,new_audio_file), format="wav")

def main(input_dir, output_dir):
    pyannote_speaker_diarlization(input_dir, output_dir)
    audio_segment(input_dir, output_dir)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-I", "--input", type=str, help="Input Directory Path")
    arg_parser.add_argument("-O", "--output", type=str, help="Output Directory Path")
    arg_parser.add_argument('-r', '--recursive', action='store_true')	# a flag that shows whether it is a recursive input folder or not
    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    is_recursive = args.recursive

    input_dir = "PAR1_Week3.wav"
    output_dir = "PAR1_Week3_diarization.csv"