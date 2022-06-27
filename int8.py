import numpy as np
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER


def calib_data():
    for _ in range(4):
        # TIP: If your calibration data is already on the GPU, you can instead provide GPU pointers
        # (as `int`s) or Polygraphy `DeviceView`s instead of NumPy arrays.
        #
        # For details on `DeviceView`, see `polygraphy/cuda/cuda.py`.
        yield {"modelInput": np.ones(shape=(1, 3, 256, 256), dtype=np.float32)}  # Totally real data


def main():
    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    calibrator = Calibrator(data_loader=calib_data(), cache="mobilevit.cache")

    # We must enable int8 mode in addition to providing the calibrator.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("mobilevit.onnx"), config=CreateConfig(int8=True, calibrator=calibrator,memory_pool_limits=200000000000)
    )
    calibrator_silu = Calibrator(data_loader=calib_data(), cache="mobilevit_silu.cache")
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("mobilevit_silu.onnx"),
        config=CreateConfig(int8=True, memory_pool_limits=200000000000,calibrator=calibrator_silu)
    )
    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    # with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
    #     # Finally, we can test out our int8 TensorRT engine with some dummy input data:
    #     inp_data = np.ones(shape=(1, 3, 256, 256), dtype=np.float32)
    #
    #     # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
    #     # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
    #     outputs = runner.infer({"modelInput": inp_data})
      #  print(outputs["modelOutput"])
       # print()
      #  assert np.array_equal(outputs["modelOutput"], inp_data)  # It's an identity model!


if __name__ == "__main__":
    main()