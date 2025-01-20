from nemo.collections.asr.models import EncDecMultiTaskModel

# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)