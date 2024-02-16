

# SAM Service

SAM Service is a Python-based web service that utilizes FastAPI and Uvicorn to create a fast and efficient API for retrieving a Segment Anything model. It also uses Nginx as a reverse proxy for handling HTTPS and improving performance. SAM stands for "Segment Anything Model," and the service is designed to make it easy for developers to generate an embedded image model for use with an ONNX runtime.

## Getting Started

Follow the steps below to run the SAM Service. Please note that these steps were performed on Ubuntu linux. Your packages and package manager commands (e.g. `apt`) may vary.

1. Make sure you have the cuda libraries installed.
```
sudo apt install nvidia-cuda-toolkit
```

1. Clone the repository: 
```
git clone git@github.com:JaneliaSciComp/SAM_service.git
```
2. Copy the model checkpoint file to the sam_service directory

 - These can be downloaded from https://github.com/facebookresearch/segment-anything#model-checkpoints
 - The `sam_vit_h_4b8939.pth` checkpoint is known to work
  
```
cp sam_vit_h_4b8939.pth SAM_service/sam_service
```

3. Install the necessary packages using conda: 
```
conda env create -f environment.yml
conda activate segment_anything
```

4. Clone the paintera-sam repo alongside this one
```
cd ..
git clone git@github.com:cmhulbert/paintera-sam.git
cd paintera-sam
pip install --user -e .
```

5. Start the API: 
```
cd sam_service
uvicorn sam_queue:app --access-log --workers 1 --host 0.0.0.0
```

Note that using one worker is very important here. Using more than one worker will spin up additional processes and each one will try to use the configured GPUs. The FAST API layer is async and doesn't require more than one worker to handle many clients.

## Deploying in Production 

### Docker

To run this service using Docker, you must first [configure Docker to work with GPUs](https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/). 

Also, you must use a recent version of docker-compose which has GPU support. The 2.24.5 version is known to work. 

The `docker-compose.yml` assumes that you put the TLS certificates in /opt/deploy/ssl. The certificate files should be named `fullchain.pem` and `privkey.pem`. 

You should also edit `nginx.conf` to set the server_name to the domain name of your server.

Finally, to start the services run the following:
```
cd docker
docker-compose up
```

Make sure Docker is enabled to restart after a reboot. 

To rebuild and push the Docker container, execute the following commands where `<version>` is the version number you want to publish:

```
docker build . -t ghcr.io/janeliascicomp/sam_service:<version>
docker push ghcr.io/janeliascicomp/sam_service:<version>
```

## Bare Metal

You can also set up everything yourself on bare metal. In production we use Nginx as a reverse proxy to handle and terminate HTTPS traffic. In this mode, Uvicorn is configured to run on a socket for improved performance. 

1. Run Uvicorn on a socket:
```
uvicorn sam_queue:app --access-log --workers 1 --forwarded-allow-ips='*' --proxy-headers --uds /tmp/uvicorn.sock
```

2. Configure nginx with the file found in `nginx.conf`

```
sudo apt-get install nginx
sudo cp nginx.conf /etc/nginx/sites-enabled/sam_service
sudo rm /etc/nginx/sites-enabled/default
sudo systemctl restart nginx
```

3. Connect to the service in your browser: `https://your-service.com/`
    - If you are running the service locally, `http://localhost:8000`

## Common Issues

### Nginx can't connect to port 80

You may have another service (like Apache) already listening on that port. Shut down that service before starting up nginx, e.g. `sudo systemctl stop apache2`

## Testing

There are scripts in the `./test` directory which can be used to verify that the service is working as intended, and to run stress tests. Use the `segment_anything` conda environment created above.

The following command starts 3 worker processes and each one submits 10 requests to the service, one at a time:
```
python tests/test_load.py -u http://localhost:8080 -i tests/em1.png -w 3 -r 10
```

This command starts 10 worker processes and each one submits 2 requests in parallel using a thread pool:
```
python tests/test_cancel.py -u http://f15u30:8000 -i tests/em1.png -w 10 -r 2 --describe
```

## Endpoint Documentation

Documentation for the endpoints is provided by the service and can be found at the `/docs` or `/redoc` URLs.

## Configuration

SAM Service can be configured by modifying the `config.json` file. The following keys

* `LOG_LEVEL`: maximum level of logging (`TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`)
* `MODEL_TYPE`: Segment Anything [model type](https://github.com/facebookresearch/segment-anything#model-checkpoints)
* `CHECKPOINT_FILE`: filename of the Segment Anything model checkpoint file
* `GPUS`: array of indicies of the GPUs to use, e.g. `[0,1,2,3]``

## Contributing

If you would like to contribute to SAM Service, feel free to open a pull request. Before doing so, please ensure that your code adheres to the PEP 8 style guide and that all tests pass.

## License

SAM Service is released under the Janelia Open-Source Software License. See `LICENSE` for more information.
