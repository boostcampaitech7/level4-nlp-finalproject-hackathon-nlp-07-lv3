import os
import json
import soundfile as sf

prefix = '/data/data_storage/'

def filter_mono_audio(json_path, output_json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_annotations = []

    for item in data['annotation']:
        audio_path = prefix + item['path']
        try:
            sr = sf.SoundFile(audio_path)  # Load audio with original channels
            if len(sr.ch) == 1:  # Mono audio (single channel)
                filtered_annotations.append(item)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    data['annotation'] = filtered_annotations

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Example usage
current_dir = os.getcwd()
json_path = os.path.join(current_dir, 'src', 'data', 'stage_2_sample_data_V2.json')
json_path_ = os.path.join(current_dir, 'src', 'data', 'stage_2_sample_data_V2_removed.json')
filter_mono_audio(json_path, json_path_)
