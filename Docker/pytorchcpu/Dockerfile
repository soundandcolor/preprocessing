from jwde/tfmididockercpu


RUN apt-get update && \
    apt-get install -yq build-essential libbz2-dev libssl-dev libreadline-dev \
                            libsqlite3-dev tk-dev && \
    apt-get install -yq libpng-dev libfreetype6-dev

#RUN cd ~ && \
#    wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh && \
#    yes | bash Anaconda3-4.3.1-Linux-x86_64.sh -b -p /opt/conda && \
#    rm Anaconda3-4.3.1-Linux-x86_64.sh

#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh

#ENV PATH /opt/conda/bin:$PATH

RUN pip install http://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp27-none-linux_x86_64.whl && \
    pip install torchvision
