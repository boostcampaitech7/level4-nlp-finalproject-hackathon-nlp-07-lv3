"""오디오 데이터 증강 및 분석을 위한 Streamlit 웹 애플리케이션."""

import streamlit as st
import librosa
import numpy as np
import os
import soundfile as sf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict
from audio_data_augmentation_by_rule import add_noise, change_speed, change_pitch, time_stretch
from dataEDA_specifirc import analyze_audio_folder
import tempfile
from zipfile import ZipFile
import io

def plot_waveform(audio: np.ndarray, sr: int, title: str = "파형") -> go.Figure:
    """
    오디오 파형을 시각화합니다.

    Args:
        audio (np.ndarray): 오디오 데이터
        sr (int): 샘플링 레이트
        title (str, optional): 그래프 제목. Defaults to "파형".

    Returns:
        go.Figure: Plotly 그래프 객체
    """
    time = np.linspace(0, len(audio) / sr, len(audio))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio, mode="lines", name="파형"))
    fig.update_layout(title=title, xaxis_title="시간 (초)", yaxis_title="진폭")
    return fig

def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "스펙트로그램") -> go.Figure:
    """
    오디오의 스펙트로그램을 시각화합니다.

    Args:
        audio (np.ndarray): 오디오 데이터
        sr (int): 샘플링 레이트
        title (str, optional): 그래프 제목. Defaults to "스펙트로그램".

    Returns:
        go.Figure: Plotly 그래프 객체
    """
    spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    fig = px.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        labels=dict(x="시간", y="주파수"),
        title=title
    )
    return fig

def create_output_directory(base_dir: str, aug_type: str, params: dict) -> str:
    """
    증강된 파일을 저장할 디렉토리를 생성합니다.

    Args:
        base_dir (str): 기본 디렉토리 경로
        aug_type (str): 증강 유형
        params (dict): 증강 파라미터

    Returns:
        str: 생성된 디렉토리 경로
    """
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # 디버그용 출력
        st.write("현재 params:", params)

        # 증강 유형별 파라미터 정보 추가
        if aug_type == "노이즈 추가":
            if "noise_factor" not in params:
                st.error("노이즈 파라미터가 전달되지 않았습니다.")
                return None
            param_info = f"noise_{params['noise_factor']:.3f}"
        elif aug_type == "시간 신축":
            if "stretch_factor" not in params:
                st.error("신축 파라미터가 전달되지 않았습니다.")
                return None
            param_info = f"stretch_{params['stretch_factor']:.1f}"
        elif aug_type == "속도 변경":
            if "speed_factor" not in params:
                st.error("속도 파라미터가 전달되지 않았습니다.")
                return None
            param_info = f"speed_{params['speed_factor']:.1f}"
        elif aug_type == "피치 변경":
            if "pitch_factor" not in params:
                st.error("피치 파라미터가 전달되지 않았습니다.")
                return None
            param_info = f"pitch_{params['pitch_factor']:+d}"
        else:
            raise ValueError(f"지원하지 않는 증강 유형: {aug_type}")

        output_dir = os.path.join(base_dir, f"augmented_{aug_type}_{param_info}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        st.write("생성된 디렉토리:", output_dir)
        return output_dir

    except Exception as e:
        st.error(f"디렉토리 생성 중 오류 발생: {str(e)}")
        return None

def get_output_filename(input_filename: str, prefix: str = "augmented") -> str:
    """
    입력 파일명을 기반으로 출력 파일명을 생성합니다.

    Args:
        input_filename (str): 입력 파일명
        prefix (str, optional): 출력 파일명 접두사. Defaults to "augmented".

    Returns:
        str: 생성된 출력 파일명
    """
    base_name, extension = os.path.splitext(input_filename)
    return f"{prefix}_{base_name}{extension}"

def process_single_audio(
    audio: np.ndarray,
    sr: int,
    aug_type: str,
    params: dict
) -> np.ndarray:
    """
    오디오 데이터를 증강 처리합니다.

    Args:
        audio (np.ndarray): 원본 오디오 데이터
        sr (int): 샘플링 레이트
        aug_type (str): 증강 유형
        params (dict): 증강 파라미터

    Returns:
        np.ndarray: 증강된 오디오 데이터
    """
    if aug_type == "노이즈 추가":
        return add_noise(audio, params["noise_factor"])
    elif aug_type == "시간 신축":
        return time_stretch(audio, params["stretch_factor"])
    elif aug_type == "속도 변경":
        return change_speed(audio, sr, params["speed_factor"])
    elif aug_type == "피치 변경":
        return change_pitch(audio, sr, params["pitch_factor"])
    return None

def process_audio_safely(
    file_data: bytes,
    aug_type: str,
    params: dict,
    original_filename: str
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    오디오 파일을 안전하게 처리합니다.

    Args:
        file_data (bytes): 오디오 파일 데이터
        aug_type (str): 증강 유형
        params (dict): 증강 파라미터
        original_filename (str): 원본 파일명

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: 증강된 오디오 데이터와 샘플링 레이트
    """
    temp_path = None
    try:
        _, extension = os.path.splitext(original_filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(file_data)
            temp_path = tmp_file.name

        audio, sr = librosa.load(temp_path, sr=None)
        augmented_audio = process_single_audio(audio, sr, aug_type, params)
        return augmented_audio, sr
    except Exception as e:
        st.error(f"오디오 처리 중 오류 발생: {str(e)}")
        return None, None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def show_comparison(
    original_audio: np.ndarray,
    augmented_audio: np.ndarray,
    sr: int
) -> None:
    """
    원본과 증강된 오디오의 특성을 비교하여 시각화합니다.

    Args:
        original_audio (np.ndarray): 원본 오디오 데이터
        augmented_audio (np.ndarray): 증강된 오디오 데이터
        sr (int): 샘플링 레이트
    """
    def calculate_features(audio_data: np.ndarray) -> dict[str, float]:
        """오디오 특성을 계산합니다."""
        return {
            "RMS 에너지": librosa.feature.rms(y=audio_data)[0].mean(),
            "제로 크로싱 레이트": librosa.feature.zero_crossing_rate(y=audio_data)[0].mean(),
            "스펙트럴 센트로이드": librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0].mean(),
            "스펙트럴 롤오프": librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0].mean(),
            "스펙트럴 대역폭": librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0].mean()
        }

    # 특성 계산
    orig_features = calculate_features(original_audio)
    aug_features = calculate_features(augmented_audio)

    # 데이터프레임 생성
    comparison_df = pd.DataFrame({
        "원본": orig_features,
        "증강": aug_features,
        "변화율(%)": {
            k: ((aug_features[k] - orig_features[k]) / orig_features[k] * 100)
            for k in orig_features.keys()
        }
    })

    # 결과 표시
    st.subheader("오디오 특성 비교")
    st.dataframe(comparison_df)

    # 특성 비교 그래프
    fig = go.Figure(data=[
        go.Bar(name="원본", x=list(orig_features.keys()), y=list(orig_features.values())),
        go.Bar(name="증강", x=list(aug_features.keys()), y=list(aug_features.values()))
    ])
    fig.update_layout(title="오디오 특성 비교", barmode="group")
    st.plotly_chart(fig)

def process_multiple_files(
    files: List,
    aug_type: str,
    params: Dict,
    output_dir: str
) -> Tuple[List[str], List[str]]:
    """
    여러 오디오 파일을 일괄 처리합니다.

    Args:
        files (List): 처리할 파일 목록
        aug_type (str): 증강 유형
        params (Dict): 증강 파라미터
        output_dir (str): 출력 디렉토리 경로

    Returns:
        Tuple[List[str], List[str]]: 처리 성공한 파일 경로 목록과 실패한 파일명 목록
    """
    total_files = len(files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    processed_files = []
    failed_files = []

    for idx, file in enumerate(files):
        status_text.text(f"처리 중... ({idx + 1}/{total_files}): {file.name}")
        try:
            augmented_audio, sr = process_audio_safely(file.getvalue(), aug_type, params, file.name)
            if augmented_audio is not None:
                output_filename = get_output_filename(file.name)
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, augmented_audio, sr)
                processed_files.append(output_path)
            else:
                failed_files.append(file.name)
        except Exception as e:
            failed_files.append(file.name)
            st.error(f"파일 처리 실패: {file.name} - {str(e)}")

        progress_bar.progress((idx + 1) / total_files)

    return processed_files, failed_files

def main() -> None:
    """
    Streamlit 웹 애플리케이션의 메인 함수입니다.
    오디오 파일의 증강 및 분석 기능을 제공합니다.
    """
    st.title("오디오 데이터 증강 및 분석 도구")

    # 저장 경로 설정
    DEFAULT_OUTPUT_PATH = "augmented_outputs"
    output_path = st.text_input("저장할 경로를 입력하세요", DEFAULT_OUTPUT_PATH)

    tab1, tab2 = st.tabs(["단일 파일 처리", "다중 파일 처리"])

    # params를 탭 밖에서 초기화
    params = {}

    with tab1:
        single_file = st.file_uploader(
            "오디오 파일을 선택하세요",
            type=["wav", "mp3", "flac"],
            key="single"
        )

        if single_file is not None:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(single_file.name)[1]
            ) as tmp_file:
                tmp_file.write(single_file.getvalue())
                temp_path = tmp_file.name

            # 오디오 로드
            audio, sr = librosa.load(temp_path, sr=None)

            st.subheader("원본 오디오")
            st.audio(single_file)
            st.plotly_chart(plot_waveform(audio, sr, "원본 파형"))
            st.plotly_chart(plot_spectrogram(audio, sr, "원본 스펙트로그램"))

            # 증강 옵션
            st.subheader("데이터 증강 옵션")
            aug_type = st.selectbox(
                "증강 유형을 선택하세요",
                ["노이즈 추가", "시간 신축", "속도 변경", "피치 변경"],
                key="single_aug_type"
            )

            if aug_type == "노이즈 추가":
                params["noise_factor"] = st.number_input(
                    "노이즈 강도 입력 (범위: 0.0 ~ 0.1)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.005,
                    step=0.001,
                    format="%.3f",
                    key="single_noise"
                )
                st.info("노이즈 강도가 클수록 더 많은 노이즈가 추가됩니다.")
            elif aug_type == "시간 신축":
                params["stretch_factor"] = st.number_input(
                    "신축 비율 입력 (범위: 0.5 ~ 2.0, 1.0=원본)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    key="single_stretch"
                )
                st.info(
                    """시간 신축은 피치를 유지하면서 오디오 길이만 변경합니다.
                    - 2.0: 오디오 길이 2배 (속도 1/2배)
                    - 0.5: 오디오 길이 1/2배 (속도 2배)"""
                )
            elif aug_type == "속도 변경":
                params['speed_factor'] = st.number_input(
                    "속도 배율 입력 (범위: 0.5 ~ 2.0, 1.0=원본)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    key="single_speed"
                )
                st.info("""속도 변경은 피치도 함께 변경됩니다.
                - 2.0: 속도 2배 빠르게 + 피치 1옥타브 상승
                - 0.5: 속도 1/2배 느리게 + 피치 1옥타브 하강""")
            elif aug_type == "피치 변경":
                params['pitch_factor'] = st.number_input(
                    "피치 변화 입력 (범위: -12 ~ +12 반음)",
                    min_value=-12,
                    max_value=12,
                    value=0,
                    step=1,
                    key="single_pitch"
                )
                st.info("양수는 피치를 올리고, 음수는 피치를 내립니다.")

            if st.button("증강 적용"):
                # 디버그용 출력 추가
                st.write("증강 적용 시 params:", params)  # 디버그용

                augmented_audio = process_single_audio(audio, sr, aug_type, params)
                output_dir = create_output_directory(output_path, aug_type, params)
                if output_dir:
                    output_filename = get_output_filename(single_file.name)
                    output_path_full = os.path.join(output_dir, output_filename)

                    sf.write(output_path_full, augmented_audio, sr)
                    st.success(f"파일이 저장되었습니다: {output_path_full}")

                    # 결과 표시
                    st.subheader("증강된 오디오")
                    st.audio(output_path_full)
                    st.plotly_chart(plot_waveform(augmented_audio, sr, "증강된 파형"))
                    st.plotly_chart(plot_spectrogram(augmented_audio, sr, "증강된 스펙트로그램"))

                    # 특성 비교
                    show_comparison(audio, augmented_audio, sr)

            # 임시 파일 삭제
            os.unlink(temp_path)

    with tab2:
        uploaded_files = st.file_uploader("오디오 파일들을 선택하세요",
                                        type=['wav', 'mp3', 'flac'],
                                        accept_multiple_files=True,
                                        key="multiple")

        if uploaded_files:
            # 파일 목록 표시
            st.write("업로드된 파일 목록:")
            file_container = st.container()
            with file_container:
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size/1024:.1f} KB)")

            st.write(f"총 파일 수: {len(uploaded_files)}")

            # 증강 옵션
            st.subheader("데이터 증강 옵션")
            aug_type = st.selectbox(
                "증강 유형을 선택하세요",
                ["노이즈 추가", "시간 신축", "속도 변경", "피치 변경"],
                key="multi_aug_type"
            )

            if aug_type == "노이즈 추가":
                params["noise_factor"] = st.number_input(
                    "노이즈 강도 입력 (범위: 0.0 ~ 0.1)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.005,
                    step=0.001,
                    format="%.3f",
                    key="multi_noise"
                )
            elif aug_type == "시간 신축":
                params["stretch_factor"] = st.number_input(
                    "신축 비율 입력 (범위: 0.5 ~ 2.0, 1.0=원본)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f"
                )
                st.info("시간 신축은 피치를 유지하면서 오디오 길이만 변경합니다.")
            elif aug_type == "속도 변경":
                params["speed_factor"] = st.number_input(
                    "속도 배율 입력 (범위: 0.5 ~ 2.0, 1.0=원본)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f"
                )
                st.info("속도 변경은 피치도 함께 변경됩니다.")
            elif aug_type == "피치 변경":
                params["pitch_factor"] = st.number_input(
                    "피치 변화 입력 (범위: -12 ~ +12 반음)",
                    min_value=-12,
                    max_value=12,
                    value=0,
                    step=1
                )

            if st.button("일괄 증강 처리"):
                output_dir = create_output_directory(output_path, aug_type, params)
                if output_dir:
                    processed_files, failed_files = process_multiple_files(
                        uploaded_files,
                        aug_type,
                        params,
                        output_dir
                    )

                    if processed_files:
                        # ZIP 파일 생성
                        zip_buffer = io.BytesIO()
                        with ZipFile(zip_buffer, "w") as zip_file:
                            for file_path in processed_files:
                                zip_file.write(file_path, os.path.basename(file_path))

                        # ZIP 파일 다운로드 버튼
                        zip_buffer.seek(0)
                        st.download_button(
                            label="증강된 파일 다운로드",
                            data=zip_buffer,
                            file_name="augmented_audio_files.zip",
                            mime="application/zip"
                        )

                    # 처리 결과 표시
                    st.success(f"처리 완료!\n- 성공: {len(processed_files)}개\n- 실패: {len(failed_files)}개")
                    if failed_files:
                        st.warning("실패한 파일들:")
                        for failed_file in failed_files:
                            st.write(f"- {failed_file}")

                    st.success(f"모든 파일이 다음 경로에 저장되었습니다: {output_dir}")

if __name__ == "__main__":
    main()
