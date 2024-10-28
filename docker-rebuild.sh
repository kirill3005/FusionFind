#!/bin/bash


docker-compose down


docker system prune -a


docker build -f Dockerfile.base -t python-app-base .


docker-compose up --build -d
