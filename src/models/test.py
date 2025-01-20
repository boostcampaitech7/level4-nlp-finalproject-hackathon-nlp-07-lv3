from nemo.collections.asr.models import EncDecMultiTaskModel

canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

print(type(canary_model.decoding))
