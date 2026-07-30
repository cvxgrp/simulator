#!/bin/bash -eu
# ClusterFuzzLite build script — installs cvx.simulator and compiles each Python
# harness in tests/fuzz/ via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC"

# Pin pip so the build environment is reproducible and only changes through a
# reviewed bump (the same rationale as the SHA-pinned base image).
pip3 install --upgrade "pip==24.3.1"

# Install the package and its runtime dependencies so PyInstaller can discover
# and bundle cvx.simulator into each frozen fuzzer binary.
pip3 install .

# PyInstaller does not discover numpy's C-extension submodules on its own, so
# the frozen fuzzer crashes at runtime with
# "No module named 'numpy._core._exceptions'". --collect-all pulls in every
# numpy submodule, data file and shared library.
#
# scipy needs the same treatment. It is not a direct dependency, but
# jquantstats 0.10.0 imports it eagerly (_portfolio_cost -> _stats._basic_core),
# so it is now on cvx.simulator's import path at fuzzer startup. Without
# --collect-all, PyInstaller misses the compiled scipy._cyutility and the frozen
# binary dies with "The `scipy` install you are using seems to be broken".
for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer" \
    --collect-all numpy \
    --collect-all pandas \
    --collect-all scipy
done
