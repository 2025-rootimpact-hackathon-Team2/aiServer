from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .utils import classify_sound, transcribe_audio

# 업로드된 오디오 파일을 처리하는 API 뷰
@api_view(["POST"])
def upload_audio(request):
    if "file" not in request.FILES:
        return Response({"error": "파일을 업로드하세요."}, status=400)
    
    file = request.FILES["file"]
    file_name = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
    file_path = default_storage.path(file_name)

    try:
        # 소리 감지
        sound_class = classify_sound(file_path)

        # 음성 인식
        transcription = transcribe_audio(file_path)

        # 결과 반환
        response_data = {
            "sound_class": sound_class,
            "transcription": transcription["text"],
            "detected_keywords": transcription["keywords"]
        }
    except Exception as e:
        return Response({"error": f"서버 오류: {str(e)}"}, status=500)
    finally:
        # 파일 삭제 (선택 사항)
        os.remove(file_path)

    return Response(response_data)
