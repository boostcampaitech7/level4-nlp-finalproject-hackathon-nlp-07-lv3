"""오디오 데이터 증강을 위한 규칙 기반 함수들을 제공하는 모듈."""

import librosa
import numpy as np
import os
import soundfile as sf
from typing import Tuple


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    오디오 파일을 로드합니다.

    Args:
        file_path (str): 오디오 파일 경로

    Returns:
        Tuple[np.ndarray, int]: 오디오 데이터와 샘플링 레이트
    """
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def save_audio(audio: np.ndarray, sr: int, file_path: str) -> None:
    """
    오디오 데이터를 파일로 저장합니다.

    Args:
        audio (np.ndarray): 오디오 데이터
        sr (int): 샘플링 레이트
        file_path (str): 저장할 파일 경로
    """
    sf.write(file_path, audio, sr)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    오디오에 가우시안 노이즈를 추가합니다.

    Args:
        audio (np.ndarray): 원본 오디오 데이터
        noise_factor (float, optional): 노이즈 강도. Defaults to 0.005.

    Returns:
        np.ndarray: 노이즈가 추가된 오디오 데이터
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio


def time_stretch(audio: np.ndarray, stretch_factor: float) -> np.ndarray:
    """
    오디오의 시간을 신축합니다 (피치 유지).

    Args:
        audio (np.ndarray): 원본 오디오 데이터
        stretch_factor (float): 신축 비율 (2.0: 2배 길게, 0.5: 2배 짧게)

    Returns:
        np.ndarray: 시간이 신축된 오디오 데이터
    """
    # stretch_factor가 2.0이면 2배 길게, 0.5면 2배 짧게 만들기 위해
    # rate 파라미터를 역수로 설정합니다.
    return librosa.effects.time_stretch(y=audio, rate=(1.0 / stretch_factor))


def change_speed(audio: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
    """
    오디오의 속도를 변경합니다 (피치도 함께 변경).

    Args:
        audio (np.ndarray): 원본 오디오 데이터
        sr (int): 샘플링 레이트
        speed_factor (float): 속도 변경 비율 (2.0: 2배 빠르게, 0.5: 2배 느리게)

    Returns:
        np.ndarray: 속도가 변경된 오디오 데이터
    """
    # 속도를 높이려면 샘플링 레이트를 낮춰야 하고,
    # 속도를 낮추려면 샘플링 레이트를 높여야 합니다.
    return librosa.resample(audio, orig_sr=sr, target_sr=int(sr / speed_factor))


def change_pitch(audio: np.ndarray, sr: int, pitch_factor: int) -> np.ndarray:
    """
    오디오의 피치를 변경합니다.

    Args:
        audio (np.ndarray): 원본 오디오 데이터
        sr (int): 샘플링 레이트
        pitch_factor (int): 피치 변경 단계 (반음 단위)

    Returns:
        np.ndarray: 피치가 변경된 오디오 데이터
    """
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)


def augment_dataset(input_dir: str, output_dir: str) -> None:
    """
    디렉토리 내의 모든 오디오 파일에 대해 증강을 수행합니다.

    Args:
        input_dir (str): 입력 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith((".wav", ".flac")):
            file_path = os.path.join(input_dir, filename)
            audio, sr = load_audio(file_path)

            # 노이즈 추가
            noisy_audio = add_noise(audio)
            save_audio(noisy_audio, sr, os.path.join(output_dir, f"noisy_{filename}"))

            # 속도 변화
            fast_audio = change_speed(audio, sr, 1.2)
            slow_audio = change_speed(audio, sr, 0.8)
            save_audio(fast_audio, sr, os.path.join(output_dir, f"fast_{filename}"))
            save_audio(slow_audio, sr, os.path.join(output_dir, f"slow_{filename}"))

            # 피치 변경
            high_pitch_audio = change_pitch(audio, sr, 2)
            low_pitch_audio = change_pitch(audio, sr, -2)
            save_audio(high_pitch_audio, sr, os.path.join(output_dir, f"high_pitch_{filename}"))
            save_audio(low_pitch_audio, sr, os.path.join(output_dir, f"low_pitch_{filename}"))


if __name__ == "__main__":
    # 사용 예시
    INPUT_DIR = "/mnt/c/naverBoostCamp/project/team2/20250119_new_repo/git_repo/src/data/LibriSpeech/train-clean-100/19/198"
    OUTPUT_DIR = "/mnt/c/naverBoostCamp/project/team2/20250119_new_repo/git_repo/src/data/augmented_LibriSpeech_train-clean-100_19_198"
    augment_dataset(INPUT_DIR, OUTPUT_DIR)

