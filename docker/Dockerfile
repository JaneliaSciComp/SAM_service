FROM condaforge/mambaforge
ARG SAM_GIT_TAG=sessions
ARG PAINTERA_SAM_GIT_HASH=b51d652

WORKDIR /app
RUN git clone --branch $SAM_GIT_TAG --depth 1 https://github.com/JaneliaSciComp/SAM_service.git
RUN git clone https://github.com/cmhulbert/paintera-sam.git \
    && cd paintera-sam \
    && git reset --hard $PAINTERA_SAM_GIT_HASH

WORKDIR /app/SAM_service
RUN mamba env create -f environment.yml \
    && mamba clean --all -y

ENTRYPOINT [ "/app/SAM_service/docker/entrypoint.sh" ]

