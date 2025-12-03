from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import os
import argparse
import time
import whisper 
import json
import re
import torch
import soundfile as sf
from pathlib import Path


def pyannote_speaker_diarlization(audio_file, output_path):
    audio_file = Path(audio_file)

    # Community-1 open-source speaker diarization pipeline
    with open("hf_token.txt", "r") as f:
        HF_TOKEN = f.read().strip()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN)

    # send pipeline to GPU (when available)
    # pipeline.to(torch.device("cuda"))
    pipeline.to(torch.device("cpu"))

    waveform_np, sample_rate = sf.read(str(audio_file)) 
    if waveform_np.ndim == 1:  # mono
        waveform_np = waveform_np[None, :]  # (1, time)
    else:  # (time, channels) -> (channels, time)
        waveform_np = waveform_np.T

    waveform = torch.from_numpy(waveform_np).float()

    # Build the in-memory audio mapping
    audio_mapping = {
        "waveform": waveform,          # (channels, time) torch.Tensor
        "sample_rate": sample_rate,    # int
        "uri": audio_file.stem,        # optional, used for naming
    }


    # apply pretrained pipeline (with optional progress hook)
    with ProgressHook() as hook:
        output = pipeline(audio_mapping, hook=hook, num_speakers=2)  # runs locally

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
    audio = AudioSegment.from_wav(str(audio_file))

    output_path = Path(output_path)
    speaker_diarlization_results_path = output_path / "speaker_diarlization_results.csv"
    speaker_diarlization_results_df = pd.read_csv(speaker_diarlization_results_path)
    speaker_summary_path = output_path / "speaker_summary.csv"
    speaker_summary_df = pd.read_csv(speaker_summary_path)

    # build mapping: speaker label -> role
    # e.g., {"SPEAKER_00": "participant", "SPEAKER_01": "interviewer"}
    speaker_to_role = dict(zip(speaker_summary_df["speaker"], speaker_summary_df["role"]))
    
    for index, row in speaker_diarlization_results_df.iterrows():
        start_time = float(row['start_time']) * 1000
        end_time = float(row['end_time']) * 1000
        speaker = row["speaker"]
        # print(str(start)+"-"+str(stop))

        # decide role directory based on speaker
        role = speaker_to_role.get(speaker, "unknown")  # fallback if not in summary
        role_dir = output_path / role / "recording_segments"
        role_dir.mkdir(parents=True, exist_ok=True)

        # create output name
        segment_file_name = f"{audio_file.stem}_{row['start_time']}-{row['end_time']}.wav"

        # cut segement
        audio_segment = audio[start_time:end_time]

        # save segment directly into output_path
        segment_path = role_dir / segment_file_name
        audio_segment.export(segment_path, format="wav")
        print(f"Saved: {segment_path}")


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
            output_file = os.path.join(output_dir, "json_version", filename + ".json")
            # output_file = os.path.join(output_dir, "json_version", filename + ".json")
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

def extract_times_from_filename(filename):
    match = re.search(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)", filename)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return start, end
    return None, None


def transfer_to_csv(output_dir):
    output_dir = Path(output_dir)

    # input/output structure
    input_dir = os.path.join(output_dir, "json_version")

    # get list of JSON files in the input directory
    json_files = [file for file in os.listdir(input_dir) if file.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    if not json_files:
        return

    # get start time, end time, and the corresponding transcript for each segment's json file
    all_full_transcripts = []  # one row per file
    for json_file in json_files:
        src_path = os.path.join(input_dir, json_file)
        print(f"Processing JSON: {src_path}")

        with open(src_path, "r", encoding="utf-8") as f:
            array = json.load(f)

        # extract start_time and end_time from filename
        start_time, end_time = extract_times_from_filename(json_file)

        # extract transcript from whisper
        full_transcript = ""
        if "text" in array:
            full_transcript = array["text"]
            all_full_transcripts.append({
                "start_time": start_time,
                "end_time": end_time,
                "full_transcript": full_transcript,
            })

    # combine all segments' transcript into once csv file (one row per file)
    if all_full_transcripts:
        full_df = pd.DataFrame(all_full_transcripts)
        full_df.to_csv(output_dir / "all_full_transcripts.csv", index=False, encoding="utf-8")
        print(f"Combined full transcripts CSV written to {output_dir / 'all_full_transcripts.csv'}")


def whisper_transcription(input_dir, output_dir, formats):
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


def main(input_dir, output_dir, formats):
    # for file in glob.glob(os.path.join(input_dir, '*.wav')):
    for file in [os.path.join(input_dir, "PAR1 W5.wav")]:
        file_basename = os.path.basename(file).split('.')[0]
        file_output_dir = os.path.join(output_dir, file_basename)
        Path(file_output_dir).mkdir(parents=True, exist_ok=True)

        pyannote_speaker_diarlization(file, file_output_dir)
        idenfity_participant(file_output_dir)
        audio_segment(file, file_output_dir)
        for role in ["interviewer", "participant"]:
            role_input_dir = os.path.join(file_output_dir, role, "recording_segments")
            role_output_dir = os.path.join(file_output_dir, role, "segment_transcripts")
            whisper_transcription(role_input_dir, role_output_dir, formats)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-I", "--input", type=str, help="Input Directory Path")
    arg_parser.add_argument("-O", "--output", type=str, help="Output Directory Path")
    arg_parser.add_argument('-r', '--recursive', action='store_true')	# a flag that shows whether it is a recursive input folder or not
    args = arg_parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    is_recursive = args.recursive
    formats = ["wav"] # change as needed

    # input_dir = "D:\\Study\\eMPowerProject\\check_in_recordings_wav_cleaned\\denoised_and_normalized"
    # output_dir = "D:\\Study\\eMPowerProject\\results"
    # is_recursive = True

    # print(torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_recursive:
        for subject in ["PAR1"]:
        # for subject in os.listdir(input_dir):
            subject_input_dir = os.path.join(input_dir, subject)

            if os.path.isdir(subject_input_dir):
                subject_outpu_dir = os.path.join(output_dir, subject)
                main(subject_input_dir, subject_outpu_dir, formats)
    else:
        main(input_dir, output_dir, formats)