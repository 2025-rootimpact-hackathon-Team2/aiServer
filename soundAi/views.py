# views.py
from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os, subprocess, logging
from .utils import convert_webm_to_wav, classify_sound, transcribe_audio

logger = logging.getLogger(__name__)

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_audio(request):
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"FILES: {request.FILES}")

    if "file" not in request.FILES:
        return Response({"error": "파일을 업로드하세요. FILES가 비어 있음"}, status=400)

    file = request.FILES["file"]
    allowed_extensions = ["wav", "mp3", "m4a", "webm", "mp4"]
    file_extension = file.name.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        return Response({"error": "지원하지 않는 파일 형식입니다."}, status=400)

    # 파일 저장
    try:
        file_name = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
        file_path = default_storage.path(file_name)
    except Exception as e:
        logger.exception("파일 저장 실패")
        return Response({"error": "파일 저장 중 오류 발생"}, status=500)

    try:
        # WebM 파일이면 WAV로 변환
        if file_extension == "webm":
            converted_file_path = file_path.replace(".webm", ".wav")
            convert_webm_to_wav(file_path, converted_file_path)
            if not os.path.exists(converted_file_path):
                raise RuntimeError("WebM→WAV 변환 후 파일이 존재하지 않음")
            file_path = converted_file_path

        # YAMNet을 이용한 소리 감지
        sound_class = classify_sound(file_path)
        if sound_class.startswith("오디오 분석 오류"):
            raise RuntimeError(sound_class)

        # Whisper를 이용한 텍스트 변환
        transcription_result = transcribe_audio(file_path)
        if transcription_result.get("error"):
            raise RuntimeError(transcription_result["error"])

        response_data = {
            "sound_class": sound_class,
            "transcription": transcription_result.get("text", ""),
            "detected_keywords": transcription_result.get("keywords", [])
        }

    except Exception as e:
        logger.exception("오디오 처리 중 오류 발생")
        return Response({"error": str(e)}, status=500)
    finally:
        logger.warning(f"끝")
        # 업로드 및 변환 파일 삭제
        # try:
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        # except Exception as e:
        #     logger.warning(f"파일 삭제 실패: {file_path}. {str(e)}")

    return Response(response_data, status=200)