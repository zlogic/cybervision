#!/bin/sh
set -e

source /etc/os-release

if [ "$ID" == "centos" ] && [ "$VERSION_ID" == "7" ]; then
    yum -y install vulkan-devel
elif [ "$ID" == "debian" ]; then
    apt-get update && apt-get install libvulkan-dev
elif [ "$ID" == "alpine" ]; then
    apk add --no-cache vulkan-loader-dev
else
    echo "Cannot detect OS from $ID"
    exit 1
fi
