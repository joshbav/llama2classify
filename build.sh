#!/bin/zsh
docker buildx build --push --platform linux/amd64 -t cr.eu-north1.nebius.cloud/e00vs3fjhrp1wapvyq/josh:7 .
