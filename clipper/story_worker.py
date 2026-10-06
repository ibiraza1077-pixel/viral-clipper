"""Runs inside .venv-story (mflux + mlx-audio live there so they can't disturb the clipper's packages).

Usage: .venv-story/bin/python clipper/story_worker.py job.json
job.json: {"tts": [{"text", "out"}], "voice", "speed",
           "images": [{"prompt", "out", "seed"}], "width", "height", "steps", "image_model"}
Prints "PROGRESS <what> <i>/<n>" lines so the main app can show progress.
Each model is loaded once per job and released before the next one, to fit in 16 GB of RAM.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
IMAGE_MODEL = "mflux-community/z-image-turbo-mflux-q4"


def say(msg: str):
    print(msg, flush=True)


def run_tts(items, voice, speed):
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    model = load_model(TTS_MODEL)
    for i, item in enumerate(items, 1):
        chunks = [np.asarray(r.audio) for r in
                  model.generate(text=item["text"], voice=voice, speed=speed, lang_code=voice[0])]
        sf.write(item["out"], np.concatenate(chunks), model.sample_rate)
        say(f"PROGRESS voice {i}/{len(items)}")
    del model
    gc.collect()


def run_images(items, width, height, steps, model_name):
    from mflux.models.common.resolution.config_resolution import ConfigResolution
    from mflux.models.z_image.variants.z_image import ZImage
    import mlx.core as mx

    # a pre-quantized repo is passed as model_path, like `mflux-generate-z-image-turbo --model <repo>` does
    model = ZImage(model_config=ConfigResolution.resolve_restricted("z-image-turbo", "z-image-turbo", model_path=model_name),
                   model_path=model_name)
    for i, item in enumerate(items, 1):
        if Path(item["out"]).exists():  # resume after a crash without redrawing
            continue
        image = model.generate_image(seed=item.get("seed", 7), prompt=item["prompt"],
                                     width=width, height=height, num_inference_steps=steps)
        image.save(path=item["out"])
        del image
        gc.collect()
        mx.clear_cache()  # stop MLX's buffer cache growing picture after picture, which slows the Mac down
        say(f"PROGRESS image {i}/{len(items)}")


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    if job.get("tts"):
        run_tts(job["tts"], job.get("voice", "bm_george"), job.get("speed", 1.1))
    if job.get("images"):
        run_images(job["images"], job.get("width", 864), job.get("height", 1080),
                   job.get("steps", 8), job.get("image_model", IMAGE_MODEL))
    say("DONE")


if __name__ == "__main__":
    main()
