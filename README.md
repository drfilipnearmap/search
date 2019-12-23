# Similarity search on prediction tiles using an auto encoder and HNSW search index
## Made by Filip Drazovic

### Instructions
1. Before starting the docker instructions, please clone spoor (https://github.com/nearmap/spoor.git) and apollo (https://github.com/nearmap/apollo.git) into modules/
2. Run the docker image 
```
docker pull nearmapltd/search:latest

docker run --name <your_container_name> --runtime=nvidia -p 5000:5000 -p 5001:5001 -it -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes -e GEN_CERT=yes -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} -v /home/NEARMAP.LOCAL/$(id -un):/home/jovyan -v /mnt:/mnt -w /home/jovyan --user root nearmapltd/search:latest start.sh /bin/bash

```

If you wish to manually build the image from the Dockerfile, run
```
docker build -f Dockerfile -t filip_search --build-arg ANACONDA_TOKEN=$ANACONDA_TOKEN .
```

3. If you're doing any model training, run a tensorboard instance 
```
tensorboard --logdir=/mnt/data/data_<your_name>/logs/ --port=5001
```
4. To run the notebooks, load Jupyter
```
jupyter lab --no-browser --port=5000 --ip=0.0.0.0
```

5. Follow instructions and code in search_full.ipynb to do everything from scratch or search_basic.ipynb to just perform a search

### Additional data

Models and indexes are available on S3 as well as /mnt/data. On S3, models are in s3://similarity-search/models/ and indexes are in s3://similarity-search/indexes/. The same files are available in /mnt/data/data_filip/models and /mnt/data/data_filip/indexes/.

Tiles and encoded predictions for the Sydney area are available in /mnt/data/data_filip/tiles_sydney/ and /mnt/data/data_filip/encoded_predictions/ respectively. 
