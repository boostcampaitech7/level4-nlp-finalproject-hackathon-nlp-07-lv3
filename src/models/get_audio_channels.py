import json
import soundfile as sf

json_path = '/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/filtered_stage2_train_sample.json'
six_channel_json_save_path = '/data/ljm/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/six.json'
# Input JSON data
data = json.load(open(json_path))

prefix = '/data/data_storage/'

n_channels = {}

# Function to get audio channel number
def get_audio_channels(file_path):
    try:
        with sf.SoundFile(prefix+file_path) as audio_file:
            return audio_file.channels

    except Exception as e:
        return f"Error: {str(e)}"

# Process each annotation and determine audio channel number
results = []
for annotation in data['annotation']:
    path = annotation['path']
    channels = get_audio_channels(path)

    # use dict
    if channels in n_channels:
        n_channels[channels] += 1
    else:
        n_channels[channels] = 1

    if channels == 6:
        # append json data
        results.append(annotation)

# Save JSON data
with open(six_channel_json_save_path, 'w') as f:
    json.dump(results, f)

# Print results
for k, v in n_channels.items():
    print(f"Audio channels: {k}, Count: {v}")
