import importlib
import os
import pathlib
from dataclasses import dataclass, field

from fastapi import FastAPI
from modal import Image, Secret, Stub, asgi_app, gpu


@dataclass(frozen=True)
class DemoApp:
    """Config object defining a ControlNet demo app's specific dependencies."""

    name: str
    model_files: list[str]
    detector_files: list[str] = field(default_factory=list)


demos = [
    DemoApp(
        name="canny2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth"
        ],
    ),
    DemoApp(
        name="depth2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
        ],
    ),
    DemoApp(
        name="fake_scribble2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        ],
    ),
    DemoApp(
        name="hed2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        ],
    ),
    DemoApp(
        name="hough2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
        ],
    ),
    DemoApp(
        name="normal2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth"
        ],
    ),
    DemoApp(
        name="pose2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
        ],
    ),
    DemoApp(
        name="scribble2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
    ),
    DemoApp(
        name="scribble2image_interactive",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth"
        ],
    ),
    DemoApp(
        name="seg2image",
        model_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth"
        ],
        detector_files=[
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"
        ],
    ),
]
demos_map: dict[str, DemoApp] = {d.name: d for d in demos}


DEMO_NAME = "scribble2image"  # Change this value to change the active demo app.
selected_demo = demos_map[DEMO_NAME]


def download_file(url: str, output_path: pathlib.Path):
    import httpx
    from tqdm import tqdm

    with open(output_path, "wb") as download_file:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            total = int(response.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        response.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = response.num_bytes_downloaded


def download_demo_files() -> None:
    """
    The ControlNet repo instructs: 'Make sure that SD models are put in "ControlNet/models".'
    'ControlNet' is just the repo root, so we place in /root/models.

    The ControlNet repo also instructs: 'Make sure that... detectors are put in "ControlNet/annotator/ckpts".'
    'ControlNet' is just the repo root, so we place in /root/annotator/ckpts.
    """
    demo = demos_map[os.environ["DEMO_NAME"]]
    models_dir = pathlib.Path("/root/models")
    for url in demo.model_files:
        filepath = pathlib.Path(url).name
        download_file(url=url, output_path=models_dir / filepath)
        print(f"download complete for {filepath}")

    detectors_dir = pathlib.Path("/root/annotator/ckpts")
    for url in demo.detector_files:
        filepath = pathlib.Path(url).name
        download_file(url=url, output_path=detectors_dir / filepath)
        print(f"download complete for {filepath}")
    print("🎉 finished baking demo file(s) into image.")


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "gradio==3.16.2",
        "albumentations==1.3.0",
        "opencv-contrib-python",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.2",
        "pytorch-lightning==1.5.0",
        "omegaconf==2.1.1",
        "test-tube>=0.7.5",
        "streamlit==1.12.1",
        "einops==0.3.0",
        "transformers==4.19.2",
        "webdataset==0.2.5",
        "kornia==0.6",
        "open_clip_torch==2.0.2",
        "invisible-watermark>=0.1.5",
        "streamlit-drawable-canvas==0.8.0",
        "torchmetrics==0.6.0",
        "timm==0.6.12",
        "addict==2.4.0",
        "yapf==0.32.0",
        "prettytable==3.6.0",
        "safetensors==0.2.7",
        "basicsr==1.4.2",
        "tqdm~=4.64.1",
    )
    # xformers library offers performance improvement.
    .pip_install("xformers", pre=True)
    .apt_install("git")
    # Here we place the latest ControlNet repository code into /root.
    # Because /root is almost empty, but not entirely empty, `git clone` won't work,
    # so this `init` then `checkout` workaround is used.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/lllyasviel/ControlNet.git",
        "cd /root && git checkout main",
    )
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .run_function(
        download_demo_files,
        secrets=[Secret.from_dict({"DEMO_NAME": DEMO_NAME})],
    )
)
stub = Stub(name="example-controlnet", image=image)

web_app = FastAPI()


def import_gradio_app_blocks(demo: DemoApp):
    from gradio import blocks

    # The ControlNet repo demo scripts are written to be run as
    # standalone scripts, and have a lot of code that executes
    # in global scope on import, including the launch of a Gradio web server.
    # We want Modal to control the Gradio web app serving, so we
    # monkeypatch the .launch() function to be a no-op.
    blocks.Blocks.launch = lambda self, server_name: print(
        "launch() has been monkeypatched to do nothing."
    )

    # each demo app module is a file like gradio_{name}.py
    module_name = f"gradio_{demo.name}"
    mod = importlib.import_module(module_name)
    blocks = mod.block
    # disable queueing mode, which is incompatible with our Modal web app setup.
    blocks.enable_queue = False
    return blocks


@stub.function(
    gpu=gpu.H100(),
    concurrency_limit=1,
    keep_warm=1,
)
@asgi_app()
def run():
    from gradio.routes import mount_gradio_app

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=import_gradio_app_blocks(demo=selected_demo),
        path="/",
    )
