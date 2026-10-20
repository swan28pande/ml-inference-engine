# syntax=docker/dockerfile:1.6
#
# CPU image: mie_server with the ONNX Runtime + reference backends.
#   docker build -t mie-server .
#   docker run --rm -p 50051:50051 mie-server
# For the CUDA / TensorRT image see docker/Dockerfile.gpu.

ARG UBUNTU_VERSION=24.04
ARG ORT_VERSION=1.22.0

# ------------------------------------------------------------------ build stage
FROM ubuntu:${UBUNTU_VERSION} AS build
ARG ORT_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git curl ca-certificates \
      libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY scripts/fetch_onnxruntime.sh scripts/
RUN ORT_VERSION=${ORT_VERSION} DEST=/opt/onnxruntime scripts/fetch_onnxruntime.sh

COPY . .
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DMIE_BUILD_TESTS=ON -DMIE_BUILD_BENCHMARKS=ON \
    && cmake --build build --parallel \
    && ctest --test-dir build --output-on-failure --timeout 120

# Collect the binaries and every shared library they need (gRPC, protobuf,
# abseil, ONNX Runtime) so the runtime image needs no -dev packages.
RUN mkdir -p /out/bin /out/lib \
    && cp build/src/mie_server build/src/mie_health_probe build/benchmarks/mie_loadgen \
          build/benchmarks/mie_engine_benchmark /out/bin/ \
    && cp -P /opt/onnxruntime/lib/libonnxruntime*.so* /out/lib/ \
    && for b in /out/bin/*; do \
         LD_LIBRARY_PATH=/opt/onnxruntime/lib ldd "$b" | awk '/=> \// {print $3}' ; \
       done | sort -u | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux)[.-]' \
       | xargs -r -I{} cp -Ln {} /out/lib/

# ---------------------------------------------------------------- runtime stage
FROM ubuntu:${UBUNTU_VERSION} AS runtime
LABEL org.opencontainers.image.title="mie-server" \
      org.opencontainers.image.description="ML inference engine: ONNX Runtime backend, dynamic batching, gRPC" \
      org.opencontainers.image.source="https://github.com/swan28pande/ml-inference-engine"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home mie

COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/lib/ /usr/local/lib/mie/
COPY configs/ /opt/mie/configs/
COPY models/sample/ /opt/mie/models/sample/
ENV LD_LIBRARY_PATH=/usr/local/lib/mie

USER mie
WORKDIR /opt/mie
EXPOSE 50051

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["mie_health_probe", "--address", "localhost:50051"]

# SIGTERM triggers a graceful drain; give it time before SIGKILL.
STOPSIGNAL SIGTERM
ENTRYPOINT ["mie_server"]
CMD ["--config", "/opt/mie/configs/server.pbtxt"]
