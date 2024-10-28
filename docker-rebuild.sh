#!/bin/bash


docker-compose down


docker system prune -a

sudo systemctl stop docker

sudo rm -rf /var/lib/docker

docker build -f Dockerfile.base -t python-app-base .


docker-compose up --build
