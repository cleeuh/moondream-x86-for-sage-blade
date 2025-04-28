# FROM waggle/plugin-base:1.1.1-base
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM waggle/plugin-base:1.1.1-ml-cuda11.0-amd64


WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get -y install --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# 11.4
# RUN python3 -m pip install --upgrade pip

RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -U pywaggle[all]
# RUN pip3 install --no-cache-dir git+https://github.com/waggle-sensor/pywaggle

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./download_model_v2.py ../
RUN cd .. && python3 download_model_v2.py
# COPY huggingface ../huggingface
# COPY huggingface /root/.cache/huggingface
COPY src .

ENTRYPOINT ["python3", "main.py"]
# ENTRYPOINT ["bash -c 'sleep 100000'"]