# QuPicT 웹 서비스 — Google Cloud Run 배포용 이미지
#
# Cloud Run이 GitHub main 브랜치에 push될 때마다 이 파일로 이미지를 만들어 다시 배포한다.
# 로컬에서 시험:  docker build -t qupict .  →  docker run --env-file .env -p 8080:8080 qupict
#
# 비밀 값(.env)은 이미지에 넣지 않는다 (.dockerignore). Cloud Run의 "변수 및 보안 비밀"에 넣는다.

FROM python:3.12-slim

# 로그가 바로바로 보이게 (버퍼링 끔), pip 캐시를 남기지 않아 이미지를 작게
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OCR(easyocr)이 쓰는 torch를 CPU판으로 먼저 설치한다.
# 그냥 두면 리눅스에서는 GPU(CUDA)판 torch가 깔려 이미지가 3GB 넘게 커진다. Cloud Run에는 GPU가 없다.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# 라이브러리 설치 (requirements.txt가 바뀌지 않으면 이 단계는 캐시를 써서 빨라진다)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 코드 복사 (.dockerignore에 적은 파일은 빠진다)
COPY . .

# Cloud Run은 PORT 환경변수(보통 8080)로 들어오는 요청을 받으라고 알려 준다.
# web.py의 main()은 DB 연결 · 크롤링 해석 프로세스를 미리 띄운 뒤 서버를 켠다.
CMD ["sh", "-c", "exec python web.py --host 0.0.0.0 --port ${PORT:-8080}"]
