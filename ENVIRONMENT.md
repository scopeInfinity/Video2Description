- Install Docker
  - `sudo apt-get install docker.io`
  - Size: 140 MB
- Install docker-composer
  - ```bash
    sudo curl -L "https://github.com/docker/compose/releases/download/1.25.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose`
    sudo chmod +x /usr/local/bin/docker-compose
    ```
  - Size: 17MB
- Test docker
  - `sudo docker run hello-world`
- Turn up new containers
  - `sudo docker-compose up`
  - Pulls image: continuumio/miniconda:4.7.12-alpine
    - Size: 48MB