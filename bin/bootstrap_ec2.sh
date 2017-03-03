#!/usr/bin/env bash

# fail script on error
set -e
set -o pipefail

# constants
rm_cmd=/bin/rm

# curret program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

function abort {
    echo "${progname}: An error occurred. Exiting..."
    exit 1
}

function install_s3 {
    # Reference doc: https://fullstacknotes.com/mount-aws-s3-bucket-to-ubuntu-file-system/
    sudo apt-get install -y build-essential git libfuse-dev libcurl4-openssl-dev libxml2-dev mime-support automake libtool
    sudo apt-get install -y pkg-config libssl-dev
    cd ~/Developer/
    git clone https://github.com/s3fs-fuse/s3fs-fuse
    cd s3fs-fuse/
    ./autogen.sh
    ./configure --prefix=/usr --with-openssl
    make
    sudo make install
}

function mount_s3 {
    mkdir -p /tmp/cache
    chmod 777 /tmp/cache
    mkdir -p ~/mnt/s3
    s3fs -oumask=0022 -o use_cache=/tmp/cache kaggle3 ~/mnt/s3
}

# setup a trip to call abort on non-zero return code
trap 'abort' 0

if which s3fs >/dev/null; then
    echo "s3fs command exists."
else
    echo "s3fs command does not exist, install it..."
    install_s3
fi

if grep -qs '/home/ubuntu/mnt/s3' /proc/mounts; then
    echo "s3 is mounted."
else
    echo "s3 is not mounted. Mount s3..."
    mount_s3
fi

# remove trap
trap : 0
