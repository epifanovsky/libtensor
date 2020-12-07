#!/bin/bash

if [ ! -f scripts/upload_to_anaconda.sh -o ! -f conda/meta.yaml.in ]; then
	echo "Please run from top dir of repository" >&2
	exit 1
fi
if [ -z "$ANACONDA_TOKEN" ]; then
	echo "Skipping build ... ANACONDA_TOKEN not set." >&2
	exit 1
fi

LTL_VERSION=$(< setup.cfg awk '/current_version/ {print; exit}' | egrep -o "[0-9.]+")
LTL_TAG=$(git tag --points-at $(git rev-parse HEAD))
if [[ "$ADCC_TAG" =~ ^v([0-9.]+)$ ]]; then
	LABEL=main
else
	LABEL=dev
	LTL_VERSION="${ADCC_VERSION}.dev"
	LTL_TAG=$(git rev-parse HEAD)
fi

echo -e "\n#"
echo "#-- Deploying tag/commit '$LTL_TAG' (version $LTL_VERSION) to label '$LABEL'"
echo -e "#\n"

set -eu

< conda/meta.yaml.in sed "s/@LTL_VERSION@/$LTL_VERSION/g;" > conda/meta.yaml

# Install requirements and setup channels
conda install conda-build anaconda-client conda-verify --yes

# Running build and deployment
conda build conda -c defaults -c conda-forge --user adcc --token $ANACONDA_TOKEN --label $LABEL
