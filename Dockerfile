FROM nearmapltd/ai_general_gpu_dev:latest

USER root

RUN apt update -yqq && \
	apt clean && \
	rm -rf /var/lib/apt/lists/*

ARG ANACONDA_TOKEN

COPY modules/spoor /spoor

RUN cd /spoor && \
	python setup.py install && \
	rm -rf /spoor

RUN conda install --force-reinstall -y conda=4.7.12 && \
	conda install -c conda-forge -c https://conda.anaconda.org/t/$ANACONDA_TOKEN/nearmap -y \
	pyutils=1.2.3 \
	pyimage_ops=1.1.27.1 \
	pygis=1.1.18 \
	pymodelstats=1.4.1 && \
	conda clean -tipsy && \
	rm -rf '~/.cache/pip/'

RUN pip install tensorflow-probability==0.7.0 \
	hnswlib==0.3.2.0 \
	nmslib==1.8.1 && \
    rm -rf '~/.cache/pip/'

USER jovyan
