from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .utils import classify_sound, transcribe_audio
import logging

logger = logging.getLogger(__name__)

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])  # multipart/form-data 지원
def upload_audio(request):
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"FILES: {request.FILES}")

    if "file" not in request.FILES:
        return Response({"error": "파일을 업로드하세요. FILES가 비어 있음"}, status=400)

    file = request.FILES["file"]

    # 파일 확장자 체크 (WAV, MP3, M4A 등만 허용)
    allowed_extensions = ["wav", "mp3", "m4a"]
    if not file.name.split(".")[-1].lower() in allowed_extensions:
        return Response({"error": "지원하지 않는 파일 형식입니다."}, status=400)

    # 파일 저장
    file_name = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
    file_path = default_storage.path(file_name)

    try:
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