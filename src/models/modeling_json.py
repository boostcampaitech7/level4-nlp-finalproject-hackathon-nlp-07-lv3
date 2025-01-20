import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

file_name = 'manifest.json'
manifest_filepath = os.path.join(current_dir, file_name)

json_string = '''{
    "train" : [
        {
            "audio_filepath": "/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/models/103-1240-0000.flac",
            "duration": 1000,
            "taskname": "asr",
            "source_lang": "en",
            "target_lang": "en",
            "pnc": "yes",
            "answer": "na"
        },
        {
            "audio_filepath": "/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/models/103-1240-0008.flac",
            "duration": 1000,
            "taskname": "asr",
            "source_lang": "en",
            "target_lang": "en",
            "pnc": "yes",
            "answer": "na"
        }
    ]
}'''

# make json
json_obj = json.loads(json_string)

# write json to file
json.dump(json_obj, open(manifest_filepath, 'w'), indent=4)
