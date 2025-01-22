import numpy as np
import soundfile as sf

# Parameters
duration = 10  # seconds
sampling_rate = 16000  # Hz
num_samples = duration * sampling_rate

# Generate random audio samples (values between -1 and 1)
audio_data = np.random.uniform(low=-1.0, high=1.0, size=num_samples).astype(np.float32)

# Save to a WAV file
output_file = "mock_audio.wav"
sf.write(output_file, audio_data, sampling_rate)

print(f"Random audio file created: {output_file}")
