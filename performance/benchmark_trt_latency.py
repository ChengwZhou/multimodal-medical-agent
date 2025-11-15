import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = os.path.abspath("./performance/exported/exported_model.plan")


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load engine")
    print(f"[INFO] Loaded engine from: {engine_path}")
    return engine


def allocate_buffers(engine, batch_size, seq_len, C, T):
    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name = "sequences"
    input_shape = (batch_size, seq_len, C, T)

    context.set_input_shape(input_name, input_shape)

    host_buffers = {}
    device_buffers = {}

    num_io = engine.num_io_tensors
    print(f"[INFO] num_io_tensors = {num_io}")

    for i in range(num_io):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)  # INPUT / OUTPUT
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        shape = context.get_tensor_shape(name)
        size = int(np.prod(shape))

        host_mem = np.empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        host_buffers[name] = (host_mem, tuple(shape))
        device_buffers[name] = device_mem

        context.set_tensor_address(name, int(device_mem))

        io_type = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"[INFO] {io_type} tensor '{name}': shape={shape}, dtype={dtype}")

    return context, stream, host_buffers, device_buffers


# === 3. 单次推理 ===
def infer_once(context, stream, host_buffers, device_buffers, input_data):
    """
    input_data: numpy array, float32, shape [B,L,C,T]
    """
    host_in, shape_in = host_buffers["sequences"]
    assert input_data.shape == shape_in, f"input shape {input_data.shape} != expected {shape_in}"
    np.copyto(host_in, input_data.ravel())

    cuda.memcpy_htod_async(device_buffers["sequences"], host_in, stream)

    context.execute_async_v3(stream_handle=stream.handle)

    host_logits, shape_logits = host_buffers["logits"]
    host_gating, shape_gating = host_buffers["gating"]

    cuda.memcpy_dtoh_async(host_logits, device_buffers["logits"], stream)
    cuda.memcpy_dtoh_async(host_gating, device_buffers["gating"], stream)

    stream.synchronize()

    logits = host_logits.reshape(shape_logits)
    gating = host_gating.reshape(shape_gating)
    return logits, gating


# === 4. 多次推理测 latency ===
def benchmark_latency(engine_path,
                      batch_size=8,
                      seq_len=10,
                      C=12,
                      T=100,
                      warmup=20,
                      iters=200):
    engine = load_engine(engine_path)
    context, stream, host_buffers, device_buffers = allocate_buffers(
        engine, batch_size=batch_size, seq_len=seq_len, C=C, T=T
    )


    dummy = np.random.randn(batch_size, seq_len, C, T).astype(np.float32)


    print(f"[INFO] Warmup {warmup} runs...")
    for _ in range(warmup):
        _ = infer_once(context, stream, host_buffers, device_buffers, dummy)


    start = cuda.Event()
    end = cuda.Event()

    print(f"[INFO] Benchmark {iters} runs...")
    start.record()
    for _ in range(iters):
        _ = infer_once(context, stream, host_buffers, device_buffers, dummy)
    end.record()
    end.synchronize()

    total_ms = start.time_till(end)  # ms
    avg_ms = total_ms / iters

    print("===================================")
    print(f"Batch   = {batch_size}")
    print(f"Seq_len = {seq_len}")
    print(f"C, T    = {C}, {T}")
    print(f"Runs    = {iters}")
    print(f"Total   = {total_ms:.3f} ms")
    print(f"Avg     = {avg_ms:.3f} ms per inference")
    print("===================================")

    return avg_ms


if __name__ == "__main__":
    benchmark_latency(
        ENGINE_PATH,
        batch_size=8,
        seq_len=10,
        C=12,
        T=100,
        warmup=20,
        iters=200,
    )
