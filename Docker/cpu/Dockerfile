FROM jwde/tfdockercpu

RUN apt-get update && \
    apt-get install -yq git && \
    apt-get install -yq vim-nox

RUN cd ~ && \
    git clone https://github.com/vishnubob/python-midi.git && \
    cd python-midi && \
    python setup.py install
