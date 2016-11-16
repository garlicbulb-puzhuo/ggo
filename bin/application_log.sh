#!/usr/bin/env bash

grep "history and metadata values" | awk -F: '{print $2}' < /dev/stdin
