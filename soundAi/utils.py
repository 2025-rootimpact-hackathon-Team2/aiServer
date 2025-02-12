import tensorflow_hub as hub
import librosa
import numpy as np
import whisper
import pandas as pd

# YAMNet 모델 로드
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Whisper 모델 로드
whisper_model = whisper.load_model("small")

# YAMNet 클래스 매핑 로컬 파일 경로 (다운로드한 CSV 파일)
class_map_path = '/home/ubuntu/django_project/aiServer/yamnet_class_map.csv'  # 여기에 실제 다운로드한 CSV 파일 경로를 입력하세요.

# CSV 파일을 로드하여 클래스 매핑
class_map = pd.read_csv(class_map_path)


# YAMNet으로 소리 감지
def classify_sound(file_path):
    try:
        # soundfile을 사용하여 오디오 파일 로드 (librosa 대신)
        audio, sr = sf.read(file_path)
        audio = audio.mean(axis=1)  # 스테레오 오디오 파일일 경우 모노로 변환
        audio = librosa.resample(audio, sr, 16000)  # 샘플링 레이트 변환 (librosa 사용)

        # YAMNet 모델 예측
        scores, embeddings, spectrogram = yamnet_model(audio)  
        
        # 가장 높은 예측 점수를 얻은 클래스 인덱스
        predicted_class = np.argmax(scores.numpy(), axis=-1)[0]  # -1 차원에서 최대값을 얻어 클래스 인덱스를 추출
        
        # 해당 클래스 번호에 해당하는 소리 종류
        predicted_class_label = class_map.iloc[predicted_class]['display_name']
        return predicted_class_label
    except Exception as e:
        raise Exception(f"Error in classify_sound: {str(e)}")

# Whisper로 한국어 텍스트 변환 및 단어 감지
def transcribe_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)  # Whisper로 텍스트 추출
        text = result["text"]  # 변환된 텍스트
        keywords = ["도와줘", "위험해", "살려줘"]  # 감지할 키워드 리스트
        detected_words = [word for word in keywords if word in text]  # 텍스트에서 키워드 검색
        return {"text": text, "keywords": detected_words}
    except Exception as e:
        raise Exception(f"Error in transcribe_audio: {str(e)}")

