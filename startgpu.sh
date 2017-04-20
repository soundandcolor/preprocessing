sudo nvidia-docker run -ti --ipc=host -p 8888:8888 -p 6006:6006 -v $PWD/..:/notebooks/sharedfolder jwde/pytorchmididockergpu bash
