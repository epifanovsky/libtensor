#!/bin/bash
set -eu

pushd $HOME  # Change to $HOME to install software

echo $OS_NAME

# Install prerequisites
if [ "$OS_NAME" == "macos-latest" ]; then
	# macOS like md5 mimics md5sum on linux
	md5sum () { command md5 -r "$@"; }

	echo ""
	echo "Installing MacOSX10.9 SDK."
	# SDK 10.9. needed for conda compilers on macOS (but then will be forward compatible)
	curl -fsSL https://github.com/phracker/MacOSX-SDKs/releases/download/10.15/MacOSX10.9.sdk.tar.xz > ~/MacOSX10.9.sdk.tar.xz
	tar -xzf ~/MacOSX10.9.sdk.tar.xz
	rm ~/MacOSX10.9.sdk.tar.xz
fi

echo
echo "Installing miniconda"
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
[ "$OS_NAME" == "macos-latest" ] && MINICONDA=Miniconda3-latest-MacOSX-x86_64.sh
MINICONDA_MD5=$(curl -s https://repo.anaconda.com/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
curl -fsSL https://repo.anaconda.com/miniconda/$MINICONDA > $MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
	echo "Miniconda MD5 mismatch"
	exit 1
fi

# Install miniconda
bash $MINICONDA -b -p "$HOME/miniconda"
rm $MINICONDA

popd  # Restore original directory
