# Local Face Estimator

로컬 FastAPI와 OpenCV DNN으로 얼굴 나이대와 성별을 추정하는 Windows 웹앱입니다.

결과는 추정치이며 실제 나이·성별을 보장하지 않습니다.

## 설치

PowerShell에서 다음을 실행합니다.

~~~powershell
cd "C:\path\to\upload"
py -m pip install -r requirements.txt
~~~

## 모델 준비

대용량 모델은 포함하지 않습니다. upload 폴더 안에 models 폴더를 만들고 다음 파일을 준비하세요.

~~~text
models/face_deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel
models/age_deploy.prototxt
models/age_net.caffemodel
models/gender_deploy.prototxt
models/gender_net.caffemodel
~~~

모델은 OpenCV DNN 얼굴 검출 샘플과 공개 Age/Gender Deep Learning 모델 저장소에서 내려받을 수 있습니다.

## 실행

~~~powershell
py -m uvicorn app:app --host 127.0.0.1 --port 8000
~~~

브라우저에서 http://127.0.0.1:8000 을 열고 JPG, PNG, WEBP 이미지를 선택합니다. 최대 크기는 10MB입니다.

API 상태 확인: GET http://127.0.0.1:8000/api/health

## 구성

app.py: FastAPI 서버와 OpenCV CPU 추론
requirements.txt: Python 의존성
static/: 업로드 화면, 미리보기, 분석 결과 UI

모델, 테스트 이미지, 캐시, 로그, 실행 중 생성 파일은 저장소에 커밋하지 않습니다.
