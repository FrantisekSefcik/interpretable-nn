# Interpretable Neural Network
Author: František Šefčík

### Instalation

1. Clone repository
```shell script
git clone git@github.com:vgg-fiit/pv-semestralny-projekt-streda-18-tamajka-FrantisekSefcik.git
```

2. Build docker image
```shell script
cd cd interpretable-nn
docker build -t interpretable-nn/tensorflow:1.15.0-gpu-py3-jupyter .
```

3. Run docker container
```shell script
cd ..
docker run --gpus all -u $(id -u):$(id -g) --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/project -it --name interpretable-nn-project interpretable-nn/tensorflow:1.15.0-gpu-py3-jupyter