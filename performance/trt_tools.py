# trt_tools.py
import os
import time
import csv

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  # 只要 import 了就会初始化 CUDA
import pynvml


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# =====================  构建 TensorRT Engine  =====================
def build_trt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_size: int = 1 << 30,  # 1GB
):
    print(f"[TRT] Building engine from: {onnx_path}")
    print(f"[TRT]  -> will save to: {engine_path}")

    # EXPLICIT_BATCH
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(flag) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # ✅ 新版 TensorRT 不再需要 / 支持 max_batch_size，直接去掉
        # builder.max_batch_size = 1

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print(f"[TRT] ERROR parsing ONNX: {onnx_path}")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        config = builder.create_builder_config()

        # ✅ 兼容旧版(max_workspace_size)和新版(set_memory_pool_limit)
        if hasattr(config, "set_memory_pool_limit"):
            # TensorRT 10+
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        elif hasattr(config, "max_workspace_size"):
            # TensorRT 8/9
            config.max_workspace_size = workspace_size
        else:
            print("[TRT] WARNING: cannot set workspace size (unknown TRT version API)")

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT]  -> Using FP16 mode")
        else:
            print("[TRT]  -> Using FP32 mode")

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            shape = list(inp.shape)

            min_shape, opt_shape, max_shape = [], [], []
            for k, d in enumerate(shape):
                if d == -1:
                    if k == 0:
                        min_shape.append(1)
                        opt_shape.append(1)
                        max_shape.append(4)
                    else:
                        min_shape.append(1)
                        opt_shape.append(4)
                        max_shape.append(8)
                else:
                    min_shape.append(d)
                    opt_shape.append(d)
                    max_shape.append(d)

            profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            print(f"[TRT]  Input '{name}' profile: "
                  f"min={min_shape}, opt={opt_shape}, max={max_shape}")

        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[TRT] ERROR: build_serialized_network returned None")
            return

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print("[TRT]  -> Done.\n")



def build_all_trt_engines(
    onnx_dir: str = "onnx_blocks",
    engine_dir: str = "trt_engines",
    fp16: bool = True,
):
    os.makedirs(engine_dir, exist_ok=True)
    onnx_files = [
        f for f in os.listdir(onnx_dir)
        if f.endswith(".onnx")
    ]
    onnx_files.sort()

    print(f"[TRT] Found {len(onnx_files)} onnx files in {onnx_dir}")
    for fname in onnx_files:
        onnx_path = os.path.join(onnx_dir, fname)
        engine_name = os.path.splitext(fname)[0] + ".plan"
        engine_path = os.path.join(engine_dir, engine_name)
        build_trt_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=fp16,
        )


# =====================  运行 Engine + 测 latency & energy  =====================

def allocate_buffers(engine: trt.ICudaEngine, batch_size: int = 1):
    context = engine.create_execution_context()

    input_host_buffers = {}
    output_host_buffers = {}
    input_device_buffers = {}
    output_device_buffers = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)  # trt.TensorIOMode.INPUT / OUTPUT

        shape = list(engine.get_tensor_shape(name))

        for j, d in enumerate(shape):
            if d == -1:
                if j == 0:
                    shape[j] = batch_size
                else:
                    # 对于其它动态维，如果有的话，给个合理的默认值
                    shape[j] = 4

        if mode == trt.TensorIOMode.INPUT:
            # 对 input 必须先告诉 context 实际 shape
            context.set_input_shape(name, tuple(shape))

        size = int(np.prod(shape))
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # 把 device 指针绑定到对应 tensor
        context.set_tensor_address(name, int(device_mem))

        if mode == trt.TensorIOMode.INPUT:
            input_host_buffers[name] = host_mem
            input_device_buffers[name] = device_mem
            print(f"[TRT] Input  tensor '{name}': shape={shape}, dtype={dtype}")
        else:
            output_host_buffers[name] = host_mem
            output_device_buffers[name] = device_mem
            print(f"[TRT] Output tensor '{name}': shape={shape}, dtype={dtype}")

    return (context,
            input_host_buffers, output_host_buffers,
            input_device_buffers, output_device_buffers)


def benchmark_engine(
    engine_path: str,
    n_warmup: int = 10,
    n_iters: int = 50,
    gpu_index: int = 0,
):
    """
    加载单个 engine，使用 IO tensor API (TensorRT 10) 做推理，
    随机输入，测平均 latency(ms) 和能量(mJ)
    """
    print(f"[BENCH] Benchmarking engine: {engine_path}")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("[BENCH]  ERROR: failed to deserialize engine.")
        return None, None

    (context,
     input_host_buffers, output_host_buffers,
     input_device_buffers, output_device_buffers) = allocate_buffers(engine)

    stream = cuda.Stream()

    # 随机初始化输入
    for name, host in input_host_buffers.items():
        host[:] = np.random.randn(host.size).astype(host.dtype)

    # NVML for energy
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    # warmup
    for _ in range(n_warmup):
        for name, host in input_host_buffers.items():
            dptr = input_device_buffers[name]
            cuda.memcpy_htod_async(dptr, host, stream)
        # TRT 10 用 execute_async_v3
        context.execute_async_v3(stream.handle)
        for name, host in output_host_buffers.items():
            dptr = output_device_buffers[name]
            cuda.memcpy_dtoh_async(host, dptr, stream)
        stream.synchronize()


    total_time = 0.0
    total_energy_mJ = 0.0

    for _ in range(n_iters):
        p_mw_before = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW

        t0 = time.perf_counter()

        for name, host in input_host_buffers.items():
            dptr = input_device_buffers[name]
            cuda.memcpy_htod_async(dptr, host, stream)

        context.execute_async_v3(stream.handle)

        for name, host in output_host_buffers.items():
            dptr = output_device_buffers[name]
            cuda.memcpy_dtoh_async(host, dptr, stream)

        stream.synchronize()
        t1 = time.perf_counter()

        dt = t1 - t0  # 秒
        total_time += dt
        total_energy_mJ += p_mw_before * dt / 1000.0  # mW * s -> mJ

    avg_latency_ms = total_time / n_iters * 1000.0
    avg_energy_mJ = total_energy_mJ / n_iters

    print(f"[BENCH]  -> avg latency: {avg_latency_ms:.3f} ms")
    print(f"[BENCH]  -> avg energy : {avg_energy_mJ:.3f} mJ\n")

    pynvml.nvmlShutdown()

    return avg_latency_ms, avg_energy_mJ
def benchmark_all_engines(
    engine_dir: str = "trt_engines",
    csv_path: str = "trt_latency_energy.csv",
    n_warmup: int = 10,
    n_iters: int = 100,
    gpu_index: int = 0,
):

    engine_files = [f for f in os.listdir(engine_dir) if f.endswith(".plan")]
    engine_files.sort()

    # 聚合结构
    results = {}

    for fname in engine_files:
        engine_path = os.path.join(engine_dir, fname)

        # 文件名格式：act_{eff_patch}_mod_{num_modal}_{block}.plan
        parts = fname.replace(".plan", "").split("_")
        eff_patch = int(parts[1])
        num_modal = int(parts[3])
        block_type = parts[4]   # agent / tokenizer / transformer

        lat, eng = benchmark_engine(
            engine_path=engine_path,
            n_warmup=n_warmup,
            n_iters=n_iters,
            gpu_index=gpu_index,
        )

        key = (num_modal, eff_patch)
        if key not in results:
            results[key] = {
                "agent": (None, None),
                "tokenizer": (None, None),
                "transformer": (None, None)
            }
        results[key][block_type] = (lat, eng)

    # ===== 写 CSV ======
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "num_modal",
            "activate_ratio",
            "patch_size",
            "T",
            "agent_ms",
            "tokenizer_ms",
            "transformer_ms",
            "total_energy_mJ",
            "N"
        ])

        for (num_modal, eff_patch), blocks in sorted(results.items()):

            # activate_ratio = eff_patch / base_patch(10)
            activate_ratio = eff_patch / 10.0

            # 你 screenshot 里 B 永远是 10
            B = 1

            # 正确的 T 计算
            T = eff_patch * 300

            agent_lat, agent_eng = blocks["agent"]
            tok_lat, tok_eng = blocks["tokenizer"]
            tr_lat, tr_eng = blocks["transformer"]

            # total energy（毫焦）
            total_energy = sum([
                e for e in [agent_eng, tok_eng, tr_eng] if e is not None
            ])

            writer.writerow([
                num_modal,
                activate_ratio,
                B,
                T,
                agent_lat or "",
                tok_lat or "",
                tr_lat or "",
                total_energy,
                n_iters
            ])

    print(f"[BENCH] Results written to {csv_path}")



# =====================  main: 先构建，再 benchmark  =====================

if __name__ == "__main__":
    onnx_dir = "onnx_blocks"      # 你保存 36 个 onnx 的目录
    engine_dir = "trt_engines"    # 保存 TensorRT engine 的目录
    fp16 = True                   # 需要 FP16 就 True


    # 2) 对所有 engine 做 latency & energy 测试
    benchmark_all_engines(
        engine_dir=engine_dir,
        csv_path="trt_latency_energy.csv",
        n_warmup=10,
        n_iters=50,
        gpu_index=0,
    )

