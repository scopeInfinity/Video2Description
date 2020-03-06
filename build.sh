#!/bin/bash
test -d "FFmpeg" || git clone 'https://github.com/FFmpeg/FFmpeg.git'
sudo docker-compose build