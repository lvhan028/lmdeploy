# How to debug Turbomind

Turbomind is implemented in C++, which is not as easy to debug as Python. This document provides basic methods for debugging Turbomind.

## Prerequisite

First, complete the local compilation according to the commands in [Install from source](../get_started/installation.md).

### Release build with detached debug symbols

By default, release packaging now strips `_turbomind`/`_xgrammar` and writes detached symbols to
`build/debug_symbols/` during CMake build.

If you need to control this behavior:

```bash
# Keep stripped release binaries and detached symbols (default in Release)
export CMAKE_BUILD_TYPE=Release
export CMAKE_ARGS="-DLMDEPLOY_SPLIT_DEBUG_INFO=ON -DLMDEPLOY_STRIP_BINARIES=ON"

# Keep host debug symbols in detached files for coredump analysis
# (default ON when LMDEPLOY_SPLIT_DEBUG_INFO=ON)
# export CMAKE_ARGS="$CMAKE_ARGS -DLMDEPLOY_EMIT_RELEASE_DEBUG_INFO=ON"

# Optional: keep CUDA source line mapping for profiler sessions
# export CMAKE_ARGS="$CMAKE_ARGS -DLMDEPLOY_ENABLE_CUDA_LINE_INFO=ON"
```

If `objcopy`/`strip` are unavailable on your system, CMake will print a warning and skip this optimization.

## Configure Python debug environment

Since many large companies currently use Centos 7 for online production environments, we will use Centos 7 as an example to illustrate the process.

### Obtain `glibc` and `python3` versions

```bash
rpm -qa | grep glibc
rpm -qa | grep python3
```

The result should be similar to this:

```
[username@hostname workdir]# rpm -qa | grep glibc
glibc-2.17-325.el7_9.x86_64
glibc-common-2.17-325.el7_9.x86_64
glibc-headers-2.17-325.el7_9.x86_64
glibc-devel-2.17-325.el7_9.x86_64

[username@hostname workdir]# rpm -qa | grep python3
python3-pip-9.0.3-8.el7.noarch
python3-rpm-macros-3-34.el7.noarch
python3-rpm-generators-6-2.el7.noarch
python3-setuptools-39.2.0-10.el7.noarch
python3-3.6.8-21.el7_9.x86_64
python3-devel-3.6.8-21.el7_9.x86_64
python3.6.4-sre-1.el6.x86_64
```

Based on the information above, we can see that the version of `glibc` is `2.17-325.el7_9.x86_64` and the version of `python3` is `3.6.8-21.el7_9.x86_64`.

### Download and install `debuginfo` library

Download `glibc-debuginfo-common-2.17-325.el7.x86_64.rpm`, `glibc-debuginfo-2.17-325.el7.x86_64.rpm`, and `python3-debuginfo-3.6.8-21.el7.x86_64.rpm` from http://debuginfo.centos.org/7/x86_64.

```bash
rpm -ivh glibc-debuginfo-common-2.17-325.el7.x86_64.rpm
rpm -ivh glibc-debuginfo-2.17-325.el7.x86_64.rpm
rpm -ivh python3-debuginfo-3.6.8-21.el7.x86_64.rpm
```

### Upgrade GDB

```bash
sudo yum install devtoolset-10 -y
echo "source scl_source enable devtoolset-10" >> ~/.bashrc
source ~/.bashrc
```

### Verification

```bash
gdb python3
```

The output should be similar to this:

```
[username@hostname workdir]# gdb python3
GNU gdb (GDB) Red Hat Enterprise Linux 9.2-10.el7
Copyright (C) 2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-redhat-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
   <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from python3...
(gdb)
```

If it shows `Reading symbols from python3`, the configuration has been successful.

For other operating systems, please refer to [DebuggingWithGdb](https://wiki.python.org/moin/DebuggingWithGdb).

## Set up symbolic links

After setting up symbolic links, there is no need to install it locally with `pip` every time.

```bash
# Change directory to lmdeploy, e.g.
cd /workdir/lmdeploy

# Since it has been built in the build directory
# Link the lib directory
cd lmdeploy && ln -s ../build/lib . && cd ..
# (Optional) Link compile_commands.json for clangd index
ln -s build/compile_commands.json .
```

## Start debugging

````bash
# Use gdb to start the API server with Llama-2-13b-chat-hf, e.g.
gdb --args python3 -m lmdeploy serve api_server /workdir/Llama-2-13b-chat-hf

# Set directories in gdb
Reading symbols from python3...
(gdb) set directories /workdir/lmdeploy

# Set a breakpoint using the relative path, e.g.
(gdb) b src/turbomind/models/llama/BlockManager.cc:104

# When it shows
# ```
# No source file named src/turbomind/models/llama/BlockManager.cc.
# Make breakpoint pending on future shared library load? (y or [n])
# ```
# Just type `y` and press enter

# Run
(gdb) r

# (Optional) Use https://github.com/InternLM/lmdeploy/blob/main/benchmark/profile_restful_api.py to send a request

python3 profile_restful_api.py --backend lmdeploy --dataset-path /workdir/ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts 1
````

## Analyze coredump with detached symbols

For stripped release binaries, keep the matching `build/debug_symbols/*.debug` files from the same build.

```bash
# Enable core dump
ulimit -c unlimited

# Example: inspect the latest core
gdb python3 /path/to/core
```

In gdb, point to detached symbols and verify build-id match:

```gdb
(gdb) set debug-file-directory /path/to/build/debug_symbols
(gdb) info sharedlibrary _turbomind
(gdb) bt
```

For address-level symbolization:

```bash
# Replace with your function address and matching .debug file
eu-addr2line -f -e /path/to/build/debug_symbols/_turbomind.cpython-*.so.debug 0xADDRESS
```

### CI recommendation

For each wheel build, archive `build/debug_symbols/` as a separate artifact (or a dedicated `-dbg` package) and
keep it versioned with wheel commit SHA/build-id. This allows postmortem debugging without shipping full symbols
inside runtime wheels.

## Using GDB

Refer to [GDB Execution Commands](https://lldb.llvm.org/use/map.html) and happy debugging.
