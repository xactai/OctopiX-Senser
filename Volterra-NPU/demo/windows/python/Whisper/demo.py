# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from datetime import datetime
from qai_hub_models.models.whisper_base_en import App as WhisperApp
from model import WhisperBaseEnONNX
import os

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="error",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file that needs to be tested. Only .mp3 are supported.",
    )
    args = parser.parse_args()
    # Input files
    encoder_path = os.path.join(ROOT_PATH, 'assets', 'models', 'WhisperEncoder.onnx')
    decoder_path = os.path.join(ROOT_PATH, 'assets', 'models', 'WhisperDecoder.onnx')

    # Load whisper model
    print("Loading model...")
    whisper = WhisperApp(WhisperBaseEnONNX(encoder_path, decoder_path))

    # Execute Whisper Model
    print("Before transcription: " + str(datetime.now().astimezone()))
    text = whisper.transcribe(os.path.join(ROOT_PATH, 'assets', 'input', args.audio_path))
    print("After transcription: " + str(datetime.now().astimezone()))
    with open(os.path.join(ROOT_PATH, 'assets', 'output', args.audio_path.replace(".\\", "").split(".")[0]+"_out.txt"), 'w') as out_file:
        out_file.write(text)
    print("After writing file: " + str(datetime.now().astimezone()))


if __name__ == "__main__":
    main()
