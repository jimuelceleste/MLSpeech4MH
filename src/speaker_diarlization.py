import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import os, glob
import argparse


def pyannote_speaker_diarlization(audio_file, output_path):
    # Community-1 open-source speaker diarization pipeline
    with open("hf_token.txt", "r") as f:
        HF_TOKEN = f.read().strip()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN)

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
            "duration": round(turn.end - turn.start, 3),
            "speaker": speaker
        })   

    # merge consecutive segments with the same speaker
    merged_segments = []
    current = segments[0]
    for segment in segments[1:]:
        if segment["speaker"] == current["speaker"]:
            current["end_time"] = segment["end_time"]
            current["duration"] += segment["duration"]
        else:
            merged_segments.append(current)
            current = segment
    merged_segments.append(current)

    # save the merged segments to csv file
    df = pd.DataFrame(merged_segments)
    df.to_csv(os.path.join(output_path, "speaker_diarlization_results.csv"), index=False)

def idenfity_participant(output_path):
    speaker_diarlization_results_df = pd.read_csv(os.path.join(output_path, "speaker_diarlization_results.csv"))
    speaker_summaries = speaker_diarlization_results_df.groupby(['speaker'])['duration'].sum()
    
    if len(speaker_summaries) == 2:
        speaker_summaries_sorted = speaker_summaries.sort_values(ascending=False)

        participant = speaker_summaries_sorted.index[0]
        interviewer = speaker_summaries_sorted.index[1]

        summary_df = pd.DataFrame({
            "speaker": speaker_summaries_sorted.index,
            "total_duration": speaker_summaries_sorted.values,
            "role": ["participant", "interviewer"]
        })
        summary_path = os.path.join(output_path, "speaker_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        return [participant, interviewer]
    else:
        print("More than 2 speakers participant identification functionality is not supported yet.")
        return [None, None]


def audio_segment(audio_file, output_path):
    audio_file = Path(audio_file)
    output_path = Path(output_path)
    csv_path = output_path / "speaker_diarlization_results.csv"
    speaker_diarlization_results_df = pd.read_csv(csv_path)
    audio = AudioSegment.from_wav(str(audio_file))
    
    for index, row in speaker_diarlization_results_df.iterrows():
        start_time = float(row['start_time']) * 1000
        end_time = float(row['end_time']) * 1000
        # print(str(start)+"-"+str(stop))

        # create output name
        new_name = f"{audio_file.stem}_{str(index + 1).zfill(3)}.wav"

        # cut segement
        audio_segment = audio[start_time:end_time]

        # Save segment directly into output_path
        segment_path = output_path / new_name
        audio_segment.export(segment_path, format="wav")
        print(f"Saved: {segment_path}")

def main(input_dir, output_dir):
    # for file in glob.glob(os.path.join(input_dir, '*.wav')):
    for file in [os.path.join(input_dir, "PAR1 W5.wav")]:
        file_basename = os.path.basename(file).split('.')[0]
        file_output_dir = os.path.join(output_dir, file_basename)
        Path(file_output_dir).mkdir(parents=True, exist_ok=True)

        # pyannote_speaker_diarlization(file, file_output_dir)
        idenfity_participant(file_output_dir)
        # audio_segment(file, file_output_dir)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-I", "--input", type=str, help="Input Directory Path")
    arg_parser.add_argument("-O", "--output", type=str, help="Output Directory Path")
    arg_parser.add_argument('-r', '--recursive', action='store_true')	# a flag that shows whether it is a recursive input folder or not
    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    is_recursive = args.recursive

    input_dir = "D:\\Study\\eMPowerProject\\check_in_recordings_wav_cleaned\\denoised_and_normalized"
    output_dir = "D:\\Study\\eMPowerProject\\results"
    is_recursive = True

    if is_recursive:
        for subject in ["PAR1"]:
        # for subject in os.listdir(input_dir):
            subject_input_dir = os.path.join(input_dir, subject)

            if os.path.isdir(subject_input_dir):
                subject_outpu_dir = os.path.join(output_dir, subject)
                main(subject_input_dir, subject_outpu_dir)
    else:
        main(input_dir, output_dir)