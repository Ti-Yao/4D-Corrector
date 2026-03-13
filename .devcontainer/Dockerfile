# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/python-3/.devcontainer/base.Dockerfile
 
# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM tensorflow/tensorflow:2.9.1-gpu
RUN apt-get install libopenexr-dev -y
RUN pip install keras==2.9.0
RUN pip install Keras-Preprocessing==1.1.2
RUN pip install matplotlib==3.5.3
RUN pip install matplotlib-inline==0.1.7
RUN pip install nibabel==5.2.1
RUN pip install numpy==1.22.4
RUN pip install oauthlib==3.2.0
RUN pip install OpenEXR==1.3.8
RUN pip install openpyxl==3.1.5
RUN pip install pandas==2.0.3
RUN pip install Pillow==9.2.0
RUN pip install protobuf==3.19.4
RUN pip install pydicom==2.2.2
RUN pip install pylibjpeg==1.4.0
RUN pip install pylibjpeg-libjpeg==1.3.4
RUN pip install pylibjpeg-openjpeg==1.3.2
RUN pip install scipy==1.9.0
RUN pip install tensorflow==2.9.1
RUN pip install tqdm==4.64.0
RUN pip install opencv-python-headless==4.7.0.68
RUN pip install ipykernel


RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*