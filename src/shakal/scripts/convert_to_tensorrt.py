#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT engines for Jetson deployment.
Run this script ON the Jetson device for optimal engine generation.
"""

import argparse
import os
import sys

def check_tensorrt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        print("TensorRT not found. Install on Jetson with:")
        print("  sudo apt-get install python3-libnvinfer python3-libnvinfer-dev")
        return None

def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True):
    trt = check_tensorrt()
    if trt is None:
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved: {engine_path}")
    print(f"Size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description='ONNX to TensorRT converter')
    parser.add_argument('onnx_path', help='Input ONNX model path')
    parser.add_argument('-o', '--output', help='Output engine path')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')

    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"File not found: {args.onnx_path}")
        return 1

    engine_path = args.output or args.onnx_path.replace('.onnx', '.engine')

    success = convert_onnx_to_tensorrt(
        args.onnx_path,
        engine_path,
        fp16=not args.fp32
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
