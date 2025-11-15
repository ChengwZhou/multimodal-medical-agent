import os
import tensorrt as trt

ONNX_PATH = os.path.abspath("./performance/exported/exported_model.onnx")
ENGINE_PATH = os.path.abspath("./performance/exported/exported_model.plan")

def build_engine_from_onnx(onnx_path, engine_path,
                           fp16=True,
                           max_workspace_size=(1 << 30)):  # 1GB
    logger = trt.Logger(trt.Logger.INFO)

    flags = 0
    if hasattr(trt, "NetworkDefinitionCreationFlag") and \
       hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
         builder.create_network(flags=flags) as network, \
         trt.OnnxParser(network, logger) as parser:

        print(f"[INFO] Loading ONNX from: {onnx_path}")
        with open(onnx_path, "rb") as f:
            model = f.read()

        print("[INFO] Parsing ONNX...")
        if not parser.parse(model):
            print("[ERROR] Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

        config = builder.create_builder_config()

        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        else:
            config.max_workspace_size = max_workspace_size

        # FP16
        if fp16 and builder.platform_has_fast_fp16:
            print("[INFO] Enabling FP16 mode")
            config.set_flag(trt.BuilderFlag.FP16)

        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        print(f"[INFO] Network input 0 name: {input_name}, shape: {input_tensor.shape}")

        C = 12
        T = 100

        profile = builder.create_optimization_profile()

        min_shape = (1, 1, C, T)
        opt_shape = (8, 10, C, T)
        max_shape = (32, 20, C, T)

        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        print("[INFO] Optimization profile set:")
        print(f"       min={min_shape}, opt={opt_shape}, max={max_shape}")

        print("[INFO] Building TensorRT engine...")

        engine_bytes = None
        if hasattr(builder, "build_serialized_network"):
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                print("[ERROR] build_serialized_network() returned None")
                return
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                print("[ERROR] build_engine() returned None")
                return
            engine_bytes = engine.serialize()

        with open(engine_path, "wb") as f:
            f.write(bytes(engine_bytes))

        print(f"[INFO] Engine saved to: {engine_path}")


if __name__ == "__main__":
    print(f"ONNX_PATH  = {ONNX_PATH}")
    print(f"ENGINE_PATH = {ENGINE_PATH}")
    build_engine_from_onnx(ONNX_PATH, ENGINE_PATH, fp16=True)
