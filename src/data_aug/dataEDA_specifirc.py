"""오디오 데이터셋의 특성을 분석하고 통계를 제공하는 모듈."""

import os
import soundfile as sf
import librosa
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import warnings
from typing import Dict, DefaultDict, List, Union, Optional

warnings.filterwarnings("ignore")


def analyze_audio_folder(folder_path: str, is_augmented: bool = False) -> Dict:
    """
    오디오 파일이 있는 폴더를 분석하여 통계 정보를 반환합니다.

    Args:
        folder_path (str): 분석할 오디오 파일이 있는 폴더 경로
        is_augmented (bool, optional): 증강된 데이터셋 여부. Defaults to False.

    Returns:
        Dict: 오디오 파일들의 통계 정보를 담은 딕셔너리
    """
    audio_stats: Dict[str, Union[int, DefaultDict, float, Dict]] = {
        "total_count": 0,
        "formats": defaultdict(int),
        "duration_sum": 0,
        "duration_min": float("inf"),
        "duration_max": float("-inf"),
        "sample_rates": defaultdict(int),
        "channels": defaultdict(int),
        "errors": defaultdict(int),
        "file_sizes": {
            "total": 0,
            "min": float("inf"),
            "max": float("-inf")
        },
        "audio_features": {
            "rms_energy": [],
            "zero_crossing_rate": [],
            "spectral_centroid": [],
            "spectral_rolloff": [],
            "spectral_bandwidth": []
        }
    }

    if is_augmented:
        audio_stats["augmentation_types"] = defaultdict(int)

    # 모든 오디오 파일 찾기
    audio_files: List[str] = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                audio_files.append(os.path.join(root, file))

    # 오디오 파일 분석
    for file_path in tqdm(audio_files, desc="오디오 파일 분석 중"):
        try:
            # 파일 크기 확인 (MB 단위)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            audio_stats["file_sizes"]["total"] += file_size
            audio_stats["file_sizes"]["min"] = min(audio_stats["file_sizes"]["min"], file_size)
            audio_stats["file_sizes"]["max"] = max(audio_stats["file_sizes"]["max"], file_size)

            # librosa를 사용하여 오디오 데이터 로드
            y, sr = librosa.load(file_path, sr=None)
            audio_stats["total_count"] += 1

            # 파일 형식
            format_ext = os.path.splitext(file_path)[1][1:].upper()
            audio_stats["formats"][format_ext] += 1

            # 길이 통계
            duration = len(y) / sr
            audio_stats["duration_sum"] += duration
            audio_stats["duration_min"] = min(audio_stats["duration_min"], duration)
            audio_stats["duration_max"] = max(audio_stats["duration_max"], duration)

            # 샘플링 레이트
            audio_stats["sample_rates"][sr] += 1

            # 오디오 특성 분석
            audio_stats["audio_features"]["rms_energy"].append(
                librosa.feature.rms(y=y)[0].mean()
            )
            audio_stats["audio_features"]["zero_crossing_rate"].append(
                librosa.feature.zero_crossing_rate(y)[0].mean()
            )
            audio_stats["audio_features"]["spectral_centroid"].append(
                librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            )
            audio_stats["audio_features"]["spectral_rolloff"].append(
                librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
            )
            audio_stats["audio_features"]["spectral_bandwidth"].append(
                librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
            )

            # 증강된 파일인 경우 증강 유형 파악
            if is_augmented:
                filename = os.path.basename(file_path)
                if filename.startswith("noisy_"):
                    audio_stats["augmentation_types"]["noise"] += 1
                elif filename.startswith("fast_"):
                    audio_stats["augmentation_types"]["speed_up"] += 1
                elif filename.startswith("slow_"):
                    audio_stats["augmentation_types"]["speed_down"] += 1
                elif filename.startswith("high_pitch_"):
                    audio_stats["augmentation_types"]["pitch_up"] += 1
                elif filename.startswith("low_pitch_"):
                    audio_stats["augmentation_types"]["pitch_down"] += 1

        except Exception as e:
            error_type = type(e).__name__
            audio_stats["errors"][error_type] += 1
            print(f"Error processing {file_path}: {str(e)}")
            continue

    return audio_stats


def print_stats(stats: Dict) -> None:
    """
    오디오 데이터셋의 통계 정보를 출력합니다.

    Args:
        stats (Dict): analyze_audio_folder 함수에서 반환된 통계 정보
    """
    print("\n=== 데이터셋 분석 결과 ===")
    print(f"총 오디오 파일 수: {stats['total_count']}")
    print(f"전체 데이터 크기: {stats['file_sizes']['total']:.2f} MB")
    print(f"파일당 평균 크기: {stats['file_sizes']['total']/stats['total_count']:.2f} MB")
    print(f"최소 파일 크기: {stats['file_sizes']['min']:.2f} MB")
    print(f"최대 파일 크기: {stats['file_sizes']['max']:.2f} MB")

    print("\n=== 오디오 특성 통계 ===")
    for feature_name, values in stats["audio_features"].items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{feature_name}:")
            print(f"  평균: {mean_val:.4f}")
            print(f"  표준편차: {std_val:.4f}")

    print("\n=== 파일 형식 분포 ===")
    for format_type, count in sorted(stats["formats"].items()):
        percentage = count / stats["total_count"] * 100
        print(f"{format_type}: {count}개 ({percentage:.1f}%)")

    print("\n=== 오디오 길이 통계 ===")
    if stats["total_count"] > 0:
        avg_duration = stats["duration_sum"] / stats["total_count"]
        print(f"평균 길이: {avg_duration:.2f}초")
        print(f"최소 길이: {stats['duration_min']:.2f}초")
        print(f"최대 길이: {stats['duration_max']:.2f}초")

    print("\n=== 샘플링 레이트 분포 ===")
    for sr, count in sorted(stats["sample_rates"].items()):
        percentage = count / stats["total_count"] * 100
        print(f"{sr} Hz: {count}개 ({percentage:.1f}%)")

    if stats["errors"]:
        print("\n=== 발생한 에러 유형 ===")
        for error_type, count in sorted(stats["errors"].items()):
            print(f"{error_type}: {count}개")


if __name__ == "__main__":
    # 경로 설정
    ORIGINAL_PATH = "/mnt/c/naverBoostCamp/project/team2/20250119_new_repo/git_repo/src/data/LibriSpeech/train-clean-100/19/198"
    AUGMENTED_PATH = "/mnt/c/naverBoostCamp/project/team2/20250119_new_repo/git_repo/src/data/augmented_LibriSpeech_train-clean-100_19_198"

    print("\n=== 원본 데이터셋 분석 ===")
    original_stats = analyze_audio_folder(ORIGINAL_PATH, is_augmented=False)
    print_stats(original_stats)

    print("\n=== 증강된 데이터셋 분석 ===")
    augmented_stats = analyze_audio_folder(AUGMENTED_PATH, is_augmented=True)
    print_stats(augmented_stats)

    # 비교 통계 출력
    print("\n=== 데이터 증강 비교 분석 ===")
    file_increase = (augmented_stats["total_count"] / original_stats["total_count"] - 1) * 100
    size_increase = (augmented_stats["file_sizes"]["total"] / original_stats["file_sizes"]["total"] - 1) * 100
    print(f"파일 수 증가율: {file_increase:.1f}%")
    print(f"전체 데이터 크기 증가율: {size_increase:.1f}%")

    # 오디오 특성 비교
    print("\n=== 오디오 특성 변화율 ===")
    for feature_name in original_stats["audio_features"].keys():
        orig_mean = np.mean(original_stats["audio_features"][feature_name])
        aug_mean = np.mean(augmented_stats["audio_features"][feature_name])
        change_rate = ((aug_mean / orig_mean) - 1) * 100
        print(f"{feature_name} 변화율: {change_rate:.1f}%")

    if "augmentation_types" in augmented_stats:
        print("\n=== 증강 유형별 분포 ===")
        for aug_type, count in augmented_stats["augmentation_types"].items():
            percentage = count / augmented_stats["total_count"] * 100
            print(f"{aug_type}: {count}개 ({percentage:.1f}%)")

