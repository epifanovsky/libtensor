#!/usr/bin/env python3
import os
import re
import sys
import subprocess
import configparser
import distutils.util

if not os.path.isfile("scripts/pack_release.py"):
    raise RuntimeError("Please run from top dir of repository")

if len(sys.argv) != 2:
    raise RuntimeError("Usage: scripts/pack_release.py install_dir")

install_dir = sys.argv[1]
if not os.path.isdir(install_dir):
    raise RuntimeError(f"Install directory {install_dir} does not exist")

platform = distutils.util.get_platform().replace('.', '_').replace('-', '_')

# Get current libtensorlight version
config = configparser.RawConfigParser()
config.read("setup.cfg")
ltl_version = config.get("bumpversion", "current_version")

# Check if this points to a tag
git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       universal_newlines=True).strip()
git_tag = subprocess.check_output(["git", "tag", "--points-at", git_revision],
                                  universal_newlines=True).strip()
if not re.match("^v([0-9.]+)$", git_tag):
    ltl_version = ltl_version + ".dev"

filename = f"libtensorlight-{ltl_version}-{platform}.tar.gz"
subprocess.check_call(["tar", "-C", install_dir, "-czf", filename, "include", "lib"])
