#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

from os.path import join

try:
    import multiprocessing

    NCPU = multiprocessing.cpu_count()
except ImportError:
    NCPU = 1


def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None)
    )


def is_ninja_available():
    status, _ = subprocess.getstatusoutput("ninja")
    return status in [0, 1]


def configure(build_dir, source_dir, install_dir, build_type=None, features=[],
              use_ninja=is_ninja_available()):
    args = ["cmake", "-DCMAKE_INSTALL_PREFIX=" + install_dir]

    if sys.platform == "darwin" and is_conda_build():
        for flag in ("LDFLAGS_LD", "LDFLAGS"):
            if flag in os.environ:
                os.environ[flag] = \
                    os.environ[flag].replace("-dead_strip_dylibs", "")

    if use_ninja:
        args += ["-G", "Ninja"]

    if build_type in ["Release", "Debug", "MinSizeRel", "RelWithDebInfo"]:
        args += ["-DCMAKE_BUILD_TYPE=" + build_type]
    elif build_type == "SanitizeAddress":
        cpflags = "-O1 -g -fsanitize=address -fno-omit-frame-pointer"
        ldflags = "-fsanitize=address"
        args += ["-DCMAKE_CXX_FLAGS_DEBUG={}".format(cpflags),
                 "-DCMAKE_C_FLAGS_DEBUG={}".format(cpflags),
                 "-DCMAKE_EXE_LINKER_FLAGS_DEBUG=\"{}\"".format(ldflags),
                 "-DCMAKE_BUILD_TYPE=Debug"]
    elif build_type:
        raise SystemExit("Unknown build type: " + build_type)

    if "mpi" in features:
        args += ["-DWITH_MPI=ON"]
    if "libxm" in features:
        args += ["-DWITH_LIBXM=ON"]
    if "mkl" in features:
        # Use sequential Intel MKL version
        args += ["-DBLA_VENDOR=Intel10_64lp_seq"]

    subprocess.check_call(args + [source_dir], cwd=build_dir)


def install(build_dir, n_jobs=NCPU, verbose=False):
    use_ninja = os.path.isfile(os.path.join(build_dir, "build.ninja"))

    args = "cmake --build . --target install --".split()
    if use_ninja:
        if verbose:
            args += ["-v"]
    else:
        args += ["-j", str(n_jobs)]
        if verbose:
            args += ["VERBOSE=1"]
    subprocess.check_call(args, cwd=build_dir)


def build_documentation(latex=True, html=True, xml=False):
    # Build documentation into the docout_dir directory
    try:
        subprocess.check_call(["doxygen", "-v"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise OSError("Doxygen not installed on the system. "
                      "Please install 'doxygen'")

    this_dir = os.path.abspath(os.path.dirname(__file__))
    subprocess.check_call("doxygen", cwd=this_dir)


def build_install(build_dir, install_dir, n_jobs=NCPU, build_type=None,
                  verbose=False, features=[]):
    install_dir = os.path.abspath(install_dir)
    build_dir = os.path.abspath(build_dir)
    source_dir = os.path.abspath(os.path.dirname(__file__))

    if os.path.isdir(build_dir):
        if not os.path.isfile(join(build_dir, "Makefile")) and \
           not os.path.isfile(join(build_dir, "build.ninja")):
            raise SystemExit("Something went wrong setting up cmake\n"
                             "Please delete " + build_dir + " and try again.")
    else:
        os.mkdir(build_dir)
        configure(build_dir, source_dir, install_dir, build_type=build_type,
                  features=features)

    install(build_dir, n_jobs=n_jobs, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Simple wrapper script around CMake to configure common "
        "build modes of libtensor."
    )

    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Run make in verbose mode.")
    parser.add_argument("--directory", "-d",  default="build", metavar="DIR",
                        help="The directory in which files are built.")
    parser.add_argument("--install", default=os.path.expanduser("~/.local/"),
                        metavar="DIR", help="The directory where files are installed.")
    parser.add_argument("--jobs", "-j", default=NCPU, metavar="N",
                        help="Number of jobs to use during build.")
    parser.add_argument("--type", "-t", default=None, metavar="BUILD_TYPE",
                        choices=["Release", "Debug", "SanitizeAddress",
                                 "MinSizeRel", "RelWithDebInfo"],
                        help="The build type to configure.")
    parser.add_argument("--features", default=[],
                        nargs="+", help="Select optional features for build.",
                        choices=["libxm", "mkl", "mpi"])
    parser.add_argument("--documentation", default=False, action="store_true",
                        help="Build documentation using doxygen.")

    args = parser.parse_args()
    build_install(args.directory, args.install, n_jobs=args.jobs,
                  build_type=args.type, verbose=args.verbose,
                  features=args.features)

    if args.documentation:
        build_documentation()


if __name__ == "__main__":
    main()
