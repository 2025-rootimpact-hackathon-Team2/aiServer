from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import subprocess
import logging
from .utils import classify_sound, transcribe_audio

logger = logging.getLogger(__name__)

def convert_webm_to_wav(input_path, output_path):
    """FFmpeg을 사용하여 WebM을 WAV로 변환"""
    command = [
        "ffmpeg",
        "-i", input_path,   # 입력 파일
        "-ar", "16000",     # 샘플 레이트 16kHz
        "-ac", "1",         # 모노 채널
        "-c:a", "pcm_s16le",  # WAV PCM 포맷
        output_path
    ]
    subprocess.run(command, check=True)

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])  # multipart/form-data 지원
def upload_audio(request):
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"FILES: {request.FILES}")

    if "file" not in request.FILES:
        return Response({"error": "파일을 업로드하세요. FILES가 비어 있음"}, status=400)

    file = request.FILES["file"]

    # 파일 확장자 체크 (WAV, MP3, M4A, WEBM만 허용)
    allowed_extensions = ["wav", "mp3", "m4a", "webm"]
    file_extension = file.name.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        return Response({"error": "지원하지 않는 파일 형식입니다."}, status=400)

    # 파일 저장
    file_name = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
    file_path = default_storage.path(file_name)

    try:
        # WebM 파일을 WAV로 변환
        if file_extension == "webm":
            converted_file_path = file_path.replace(".webm", ".wav")
            convert_webm_to_wav(file_path, converted_file_path)
            file_path = converted_file_path  # 변환된 파일 사용

        # 🔹 YAMNet을 이용한 소리 감지
        sound_class = classify_sound(file_path)

        # 🔹 Whisper를 이용한 텍스트 변환
        transcription = transcribe_audio(file_path)

        # 🔹 오류 체크
        if "error" in transcription:
            return Response({"error": transcription["error"]}, status=500)

        # 🔹 결과 데이터 생성
        response_data = {
            "sound_class": sound_class,
            "transcription": transcription["text"],
            "detected_keywords": transcription["keywords"]
        }

    except Exception as e:
        return Response({"error": str(e)}, status=500)

    finally:
        # 업로드된 파일 삭제 (필요시 주석 처리 가능)
        if os.path.exists(file_path):
            os.remove(file_path)

    return Response(response_data, status=200)