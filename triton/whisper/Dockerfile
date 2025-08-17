# Build: sudo docker build --force-rm -t newbing/xwhisper-tensorrt:latest .
# Run: sudo docker run --rm --gpus all -p 8080:8080 -e TRITON_SERVER_API_PORT=127.0.0.1:8001 newbing/xwhisper-tensorrt:latest
# Step 1：Build
FROM soar97/triton-whisper:24.09 as builder

ENV MODEL_ID=large-v3-turbo
ENV PYTHONIOENCODING=utf-8

COPY prepare.sh .

RUN /bin/bash ./prepare.sh large-v3-turbo

# Step 2：Running
FROM soar97/triton-whisper:24.09
COPY --from=builder /workspace/model_repo_whisper /workspace/model_repo_whisper
COPY --from=builder /workspace/whisper_large-v3-turbo /workspace/whisper_large-v3-turbo
COPY main.py /workspace
COPY requirements.txt /workspace
RUN apt update \
    && apt install -y supervisor tree \
    && tree /workspace/model_repo_whisper \
    && tree /workspace/whisper_large-v3-turbo \
    && python3 -m pip install -r requirements.txt\
    && ls -ahl /etc/supervisor \
    && ls -ahl /etc/supervisor/conf.d/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV PYTHONIOENCODING=utf-8

CMD ["/usr/bin/supervisord"]