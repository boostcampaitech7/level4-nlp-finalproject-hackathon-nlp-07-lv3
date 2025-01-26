import os
import json
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

src_path = os.path.join(current_dir, 'data', 'test_aac.json')
dst_path = os.path.join(current_dir, 'data', 'test_aac.json')

default_manifest_options = {
    "duration": 1500,"source_lang": "en","target_lang": "en","pnc": "yes","answer": "na"
}

manifest_prefix = '/data/data_storage'

def json_to_manifest(src_path, dst_path):
    # read each path in annotation section of src json file
    # write each path to dst jsonl file
    open(dst_path, 'w').close()

    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ex) data[0]['path'] = 'data/whisper/whisper_00000.wav'
    # then jsonl line should be {"audio_filepath": "data/whisper/whisper_00000.wav","duration": 1500,"taskname": "asr","source_lang": "en","target_lang": "en","pnc": "yes","answer": "na"}
    with open(dst_path, 'w', encoding='utf-8') as f:
        for d in data['annotation']:
            audio_filepath = d['path']
            slash = '' if audio_filepath[0] == '/' else '/'
            audio_filepath = manifest_prefix + slash + audio_filepath

            f.write(json.dumps({"audio_filepath": audio_filepath, "taskname": "asr", **default_manifest_options}) + '\n')

    return dst_path

def json_to_manifest_indice(src_path, dst_path, indices):
    # erase all data in dst_path
    open(dst_path, 'w').close()

    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(dst_path, 'w', encoding='utf-8') as f:
        for i in indices:
            d = data['annotation'][i]
            audio_filepath = d['path']
            slash = '' if audio_filepath[0] == '/' else '/'
            audio_filepath = manifest_prefix + slash + audio_filepath

            task_name = d['task']

            f.write(json.dumps({"audio_filepath": audio_filepath, "taskname": "asr", **default_manifest_options}) + '\n')

    return dst_path

if __name__ == "__main__":
    json_to_manifest(src_path, dst_path)
