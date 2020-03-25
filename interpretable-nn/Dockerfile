FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN mkdir project

WORKDIR /project/

COPY ./requirements.txt /project/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root