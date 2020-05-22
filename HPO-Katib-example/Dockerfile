FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y wget python3-dev gcc && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py


RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install matplotlib
COPY requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

ADD . /opt/jhs_keras
WORKDIR /opt/jhs_keras


ENTRYPOINT ["python3", "/opt/jhs_keras/CNN_MNIST_CUSTOM_VGG19.py"]