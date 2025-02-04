import json
import soundfile as sf
from tqdm import tqdm

json_path = '/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/test_aac.json'
_json_save_path = '/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/stage_2_sample_data_V2_remove.json'
# Input JSON data
data = json.load(open(json_path))

prefix = '/data/data_storage/'

n_channels = {}

# Function to get audio channel number
def get_audio_channels(file_path):

    try:
        audio, sr = sf.read(prefix+file_path)
        return audio, sr

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(0)

#Process each annotation and determine audio channel number
results = []
for annotation in tqdm(data['annotation']):
    path = annotation['path']
    try:
        audio, sr = get_audio_channels(path)

    except Exception as e:
        print(f"error occurred : {e}")

# output_data = {"annotation": results}
# # Save JSON data
# with open(_json_save_path, 'w') as f:
#     json.dump(output_data, f)
