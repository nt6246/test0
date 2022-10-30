from utils.ncnn_model import NcnnModel, NcnnModelWrapper
import argparse

if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        prog="ESRGAN MODEL SCALE CHECKER (NCNN ONLY)",
        description="Code from ChaiNNer. Modified by DrPleaseRespect",
        epilog="ChaiNNer: https://github.com/chaiNNer-org/chaiNNer DrPleaseRespect: https://github.com/DrPleaseRespect "
    )
    parser.add_argument("bin")
    parser.add_argument("param")
    args = parser.parse_args()
    model_scale = NcnnModelWrapper(
        NcnnModel.load_from_file(args.param, args.bin)
    ).scale
    print(F"Scale: {model_scale}")