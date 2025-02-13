# utils.py
import tensorflow_hub as hub
import librosa
import numpy as np
import whisper
import pandas as pd
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

# YAMNet 모델 로드
try:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
except Exception as e:
    logger.exception("YAMNet 모델 로드 실패")

# Whisper 모델 로드
try:
    whisper_model = whisper.load_model("small")
except Exception as e:
    logger.exception("Whisper 모델 로드 실패")

# CSV 파일 경로: 서버 환경에 맞게 절대 경로를 설정하세요.
# 예시: 환경변수나 settings에서 경로를 가져오도록 할 수도 있음.
class_map_path = os.environ.get("YAMNET_CLASS_MAP", "/home/ubuntu/django-server/aiServer/yamnet_class_map.csv")

try:
    class_map = pd.read_csv(class_map_path)
except Exception as e:
    logger.exception(f"YAMNet 클래스 매핑 CSV 로드 실패: {class_map_path}")
    class_map = None

def convert_webm_to_wav(input_path, output_path):
    """
    FFmpeg을 사용하여 WebM 파일을 16kHz, 모노 WAV로 변환.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",  # 기존 파일 덮어쓰기
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"WebM → WAV 변환 성공: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 변환 실패: {e.stderr.decode('utf-8')}")
        raise

def classify_sound(file_path):
    """
    YAMNet을 이용해 오디오 파일을 분석하고, 가장 높은 점수를 받은 소리 클래스 라벨을 반환.
    """
    try:
        # librosa가 PySoundFile 실패 시 audioread로 로드하므로 경고는 무시
        audio, sr = librosa.load(file_path, sr=16000)
        if audio is None or len(audio) == 0:
            raise ValueError("오디오 데이터가 비어 있음")
        # YAMNet 모델에 입력할 때는 float32 타입이 필요함
        audio = audio.astype('float32')
        scores, embeddings, spectrogram = yamnet_model(audio)
        # scores는 (frames, classes) 형태. 전체 평균 점수를 계산하여 최고 클래스를 선택
        mean_scores = np.mean(scores.numpy(), axis=0)
        predicted_class = np.argmax(mean_scores)
        if class_map is None:
            return f"클래스 매핑 파일 오류: {class_map_path}"
        predicted_class_label = class_map.iloc[predicted_class]['display_name']
        return predicted_class_label
    except Exception as e:
        logger.exception("오디오 분석 오류")
        return f"오디오 분석 오류: {str(e)}"

def transcribe_audio(file_path):
    """
    Whisper 모델을 이용해 오디오 파일에서 텍스트를 추출하고,
    특정 키워드를 감지하여 반환.
    """
    try:
        result = whisper_model.transcribe(file_path)
        text = result.get("text", "")
        keywords = ["도와줘", "위험해", "살려줘"]
        detected_words = [word for word in keywords if word in text]
        return {"text": text, "keywords": detected_words}
    except Exception as e:
        logger.exception("음성 인식 오류")
        return {"text": "", "keywords": [], "error": f"음성 인식 오류: {str(e)}"}