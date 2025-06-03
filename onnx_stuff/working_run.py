#from quark.onnx.quantization.config.config import Config, QuantizationConfig
from quark.onnx.quantization.config import (Config, get_default_config)
import onnxruntime as ort
import quark
from quark.onnx import get_library_path as vai_lib_path
from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
from quark.onnx.quantization.config.config import Config, QuantizationConfig
from quark.onnx import ModelQuantizer, VitisQuantType, VitisQuantFormat
from onnxruntime.quantization.calibrate import CalibrationMethod
import onnx


#g_config = get_default_config("BF16")
g_config = get_default_config("XINT8")

quant_config = QuantizationConfig(
    calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
    quant_format=quark.onnx.QuantFormat.QDQ,
    execution_providers=['ROCMExecutionProvider']
)

#quant_config = QuantizationConfig(
#                                  calibrate_method=CalibrationMethod.MinMax,
#                                  #calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
#                                  #quant_format=VitisQuantFormat.QDQ,
#                                  #activation_type=VitisQuantType.QBFloat16,
#                                  #weight_type=VitisQuantType.QBFloat16,
#                                  #extra_options={'BF16QDQToCast': True},
#                                  execution_providers=['ROCMExecutionProvider']
#                                  )

#quant_config = QuantizationConfig(
    #calibrate_method=CalibrationMethod.MinMax,
    #quant_format=quark.onnx.VitisQuantFormat.BFPFixNeuron,
    #activation_type=quark.onnx.VitisQuantType.QBFP,
    #weight_type=quark.onnx.VitisQuantType.QBFP,
#)



config = Config(global_quant_config=quant_config)

config.global_quant_config.extra_options["UseRandomData"] = True
print("The configuration of the quantization is {}".format(config))


import os

so = ort.SessionOptions()
so.register_custom_ops_library(vai_lib_path('cuda'))

fpath = "/home/schoch/.cache/rtmlib/hub/checkpoints/"
#fname = "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
fname = "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx"
pPath = os.path.join(fpath + fname)

model = onnx.load(pPath)

output_model_path = "testopt_halpe26.onnx"
input_model_path = pPath
#quantizer = ModelQuantizer(quant_config)
quantizer = ModelQuantizer(config)
quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)

#fname = "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
pPath = os.path.join(fpath + fname)

model = onnx.load(pPath)

output_model_path = "testopt_det.onnx"
input_model_path = pPath
#quantizer = ModelQuantizer(quant_config)
quantizer = ModelQuantizer(config)
quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)

