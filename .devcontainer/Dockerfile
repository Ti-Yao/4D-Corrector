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
RUN pip install scikit-image

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*