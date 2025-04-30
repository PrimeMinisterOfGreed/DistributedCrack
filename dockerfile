FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 as base 
# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    libssl-dev \
    curl \
    libopenmpi-dev \
    openmpi-common \
    nvidia-cuda-toolkit \
    gcc-12 \
    g++-12 \
    clang \
    libclang-dev \
    python3 \
    cmake 

RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH="~/.cargo/bin:${PATH}"
RUN ln -s ~/.cargo/bin/rustup /usr/local/bin/rustup
RUN ln -s ~/.cargo/bin/cargo /usr/local/bin/cargo
# add x86_64-unknown-linux-gnu target
# Copy workspace
WORKDIR /app
COPY  . .
# Set the working directory
RUN  mkdir -p  /app/build && \
    cd /app/build && \
    cmake .. && \
    cmake --build . && \
    cmake --build . --target md5 && \
    cmake --build . --target mpc

RUN ls ./build/core/release
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 as runtime
# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    libssl-dev \
    libopenmpi-dev \
    openmpi-common \
    python3 \
    cmake

# Copy the built files from the builder stage
COPY --from=base /app/build/core/release/mpc /usr/local/bin/mpc
# Copy launch.toml from the main host
RUN mkdir -p /config 

COPY launch.toml /config/launch.toml

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/mpc"]