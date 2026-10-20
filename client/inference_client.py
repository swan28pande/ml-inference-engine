#!/usr/bin/env python3
"""Python client for mie_server.

    pip install -r client/requirements.txt
    python client/inference_client.py health
    python client/inference_client.py metadata --model mlp
    python client/inference_client.py predict --model mlp --batch 4
    python client/inference_client.py split --model mlp --weights 1=50,2=50
    python client/inference_client.py load --model mlp --concurrency 32 --duration 10
    python client/inference_client.py metrics --prometheus
    python client/inference_client.py demo          # runs all of the above

Protobuf stubs are generated from ../proto on first use (grpcio-tools).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from collections import Counter

import grpc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "generated")
PROTO_DIR = os.environ.get("MIE_PROTO_DIR", os.path.join(HERE, "..", "proto"))


def _load_stubs():
    """Imports the generated stubs, generating them from the .proto if needed."""
    if not os.path.exists(os.path.join(GENERATED, "inference_pb2_grpc.py")):
        from grpc_tools import protoc

        os.makedirs(GENERATED, exist_ok=True)
        rc = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_DIR}",
                f"--python_out={GENERATED}",
                f"--grpc_python_out={GENERATED}",
                os.path.join(PROTO_DIR, "inference.proto"),
            ]
        )
        if rc != 0:
            raise RuntimeError(f"protoc failed with exit code {rc}")
    sys.path.insert(0, GENERATED)
    import inference_pb2  # noqa: E402
    import inference_pb2_grpc  # noqa: E402

    return inference_pb2, inference_pb2_grpc


pb, pb_grpc = _load_stubs()

_NP_TO_PB = {
    np.dtype("float32"): pb.DT_FLOAT32,
    np.dtype("float16"): pb.DT_FLOAT16,
    np.dtype("int64"): pb.DT_INT64,
    np.dtype("int32"): pb.DT_INT32,
    np.dtype("uint8"): pb.DT_UINT8,
    np.dtype("bool"): pb.DT_BOOL,
}
_PB_TO_NP = {v: k for k, v in _NP_TO_PB.items()}


def to_tensor(name: str, array: np.ndarray) -> "pb.Tensor":
    array = np.ascontiguousarray(array)
    return pb.Tensor(name=name, dtype=_NP_TO_PB[array.dtype], shape=list(array.shape), raw_data=array.tobytes())


def from_tensor(t: "pb.Tensor") -> np.ndarray:
    return np.frombuffer(t.raw_data, dtype=_PB_TO_NP[t.dtype]).reshape(tuple(t.shape))


class InferenceClient:
    def __init__(self, address: str = "localhost:50051", timeout: float = 5.0):
        self.channel = grpc.insecure_channel(
            address,
            options=[("grpc.max_receive_message_length", 64 << 20), ("grpc.max_send_message_length", 64 << 20)],
        )
        self.stub = pb_grpc.InferenceServiceStub(self.channel)
        self.timeout = timeout

    def wait_ready(self, timeout: float = 30.0) -> None:
        grpc.channel_ready_future(self.channel).result(timeout=timeout)

    def health(self):
        return self.stub.Health(pb.HealthRequest(), timeout=self.timeout)

    def metadata(self, model: str, version: str = ""):
        return self.stub.GetModelMetadata(pb.ModelMetadataRequest(model_name=model, model_version=version),
                                          timeout=self.timeout)

    def list_models(self):
        return self.stub.ListModels(pb.ListModelsRequest(), timeout=self.timeout)

    def predict(self, model: str, inputs: dict, version: str = "", routing_key: str = "", outputs=None):
        req = pb.PredictRequest(model_name=model, model_version=version, routing_key=routing_key,
                                inputs=[to_tensor(k, v) for k, v in inputs.items()], output_filter=outputs or [])
        resp = self.stub.Predict(req, timeout=self.timeout)
        return {t.name: from_tensor(t) for t in resp.outputs}, resp

    def set_split(self, model: str, weights: dict):
        return self.stub.SetTrafficSplit(pb.SetTrafficSplitRequest(model_name=model, weights=weights),
                                         timeout=self.timeout)

    def metrics(self, model: str = "", prometheus: bool = False):
        return self.stub.GetMetrics(pb.MetricsRequest(model_name=model, prometheus_format=prometheus),
                                    timeout=self.timeout)

    def random_inputs(self, model: str, batch: int = 1) -> dict:
        md = self.metadata(model)
        inputs = {}
        for spec in md.inputs:
            shape = [batch if (d < 0 and i == 0) else (1 if d < 0 else d) for i, d in enumerate(spec.shape)]
            inputs[spec.name] = np.random.standard_normal(shape).astype(_PB_TO_NP[spec.dtype])
        return inputs


# --------------------------------------------------------------------- commands

def cmd_health(c: InferenceClient, _args) -> None:
    h = c.health()
    print(f"status={pb.ServingStatus.Name(h.status)} uptime={h.uptime_seconds}s in_flight={h.in_flight_requests} "
          f"version={h.server_version}")
    for m in h.models:
        print(f"  {m.model_name}:{m.version:<4} {pb.ModelState.Name(m.state):<9} backend={m.backend} "
              f"queue={m.queue_depth} {m.message}")
    for g in h.gpus:
        print(f"  gpu{g.device_id} {g.name} util={g.utilization_percent:.0f}% "
              f"mem={g.memory_used_bytes / 2**30:.1f}/{g.memory_total_bytes / 2**30:.1f} GiB")
    if not h.gpus:
        print("  (no GPUs reported)")


def cmd_metadata(c: InferenceClient, args) -> None:
    md = c.metadata(args.model, args.version)
    print(f"{md.model_name}:{md.version} backend={md.backend} platform={md.platform} max_batch={md.max_batch_size}")
    for kind, specs in (("input", md.inputs), ("output", md.outputs)):
        for s in specs:
            print(f"  {kind:<6} {s.name:<16} {pb.DataType.Name(s.dtype):<11} {list(s.shape)}")


def cmd_list(c: InferenceClient, _args) -> None:
    for m in c.list_models().models:
        weights = dict(m.traffic_weights) or "latest"
        print(f"{m.model_name}: versions={[v.version for v in m.versions]} traffic={weights}")


def cmd_predict(c: InferenceClient, args) -> None:
    outputs, resp = c.predict(args.model, c.random_inputs(args.model, args.batch), version=args.version,
                              routing_key=args.key)
    t = resp.timing
    print(f"served by {resp.model_name}:{resp.model_version}  total={t.total_us}us queue={t.queue_us}us "
          f"compute={t.compute_us}us executed_batch={t.batch_size}")
    for name, arr in outputs.items():
        print(f"  {name} {arr.shape}\n{np.array2string(arr, precision=4, suppress_small=True)}")


def cmd_split(c: InferenceClient, args) -> None:
    weights = {}
    if args.weights:
        for item in args.weights.split(","):
            version, weight = item.split("=")
            weights[version] = int(weight)
    resp = c.set_split(args.model, weights)
    print(f"traffic split for {args.model}: {dict(resp.weights) or 'cleared (latest version)'}")


def cmd_metrics(c: InferenceClient, args) -> None:
    resp = c.metrics(args.model, args.prometheus)
    for m in resp.models:
        lat = m.request_latency
        print(f"{m.model_name}:{m.version} ok={m.requests_ok} failed={m.requests_failed} "
              f"rejected={m.requests_rejected} rps={m.throughput_rps:.0f} mean_batch={m.mean_batch_size:.1f} "
              f"p50={lat.p50_us / 1000:.2f}ms p95={lat.p95_us / 1000:.2f}ms p99={lat.p99_us / 1000:.2f}ms")
    if args.prometheus:
        print(resp.prometheus_text)


def cmd_load(c: InferenceClient, args) -> None:
    inputs = c.random_inputs(args.model, args.batch)
    request = pb.PredictRequest(model_name=args.model, model_version=args.version,
                                inputs=[to_tensor(k, v) for k, v in inputs.items()])
    latencies, versions, errors = [], Counter(), Counter()
    lock = threading.Lock()
    deadline = time.perf_counter() + args.duration

    def worker(i: int) -> None:
        req = pb.PredictRequest()
        req.CopyFrom(request)
        if args.key:
            req.routing_key = f"{args.key}-{i}"  # sticky per worker
        local, local_versions = [], Counter()
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                resp = c.stub.Predict(req, timeout=c.timeout)
                local.append((time.perf_counter() - t0) * 1000)
                local_versions[resp.model_version] += 1
            except grpc.RpcError as e:
                with lock:
                    errors[e.code().name] += 1
        with lock:
            latencies.extend(local)
            versions.update(local_versions)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    if not latencies:
        print(f"no successful requests; errors={dict(errors)}")
        sys.exit(1)
    q = statistics.quantiles(latencies, n=100)
    print(f"requests={len(latencies)} throughput={len(latencies) / elapsed:.0f} req/s "
          f"p50={q[49]:.2f}ms p95={q[94]:.2f}ms p99={q[98]:.2f}ms errors={dict(errors) or 0}")
    print(f"version split: {dict(versions)}")
    print("(the Python client is GIL-bound; use mie_loadgen for server capacity measurements)")


def cmd_demo(c: InferenceClient, args) -> None:
    print("== health");   cmd_health(c, args)
    print("\n== models");  cmd_list(c, args)
    print("\n== metadata"); cmd_metadata(c, args)
    print("\n== predict");  cmd_predict(c, args)
    print("\n== load (A/B split as configured)")
    args.duration = min(args.duration, 5)
    cmd_load(c, args)
    print("\n== metrics");  cmd_metrics(c, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["health", "metadata", "list", "predict", "split", "metrics", "load", "demo"])
    parser.add_argument("--address", default=os.environ.get("MIE_SERVER", "localhost:50051"))
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--version", default="")
    parser.add_argument("--key", default="", help="sticky routing key")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--weights", default="", help="e.g. 1=90,2=10 (empty clears the split)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--prometheus", action="store_true")
    parser.add_argument("--wait", type=float, default=30, help="seconds to wait for the server")
    args = parser.parse_args()

    client = InferenceClient(args.address)
    client.wait_ready(args.wait)
    commands = {
        "health": cmd_health, "metadata": cmd_metadata, "list": cmd_list, "predict": cmd_predict,
        "split": cmd_split, "metrics": cmd_metrics, "load": cmd_load, "demo": cmd_demo,
    }
    try:
        commands[args.command](client, args)
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code().name}: {e.details()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
