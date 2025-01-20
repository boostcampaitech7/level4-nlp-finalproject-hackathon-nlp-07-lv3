import os
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name="nvidia/canary-1b")
preprocessor = nemo_asr.modules.AudioToMelSpectrogramPreprocessor.from_config_dict(asr_model._cfg.preprocessor)

audio_path = os.path.join(os.getcwd(), 'src', 'models', '103-1240-0000.flac')
spectogram = preprocessor.process_from_file(audio_path)

print(spectogram)