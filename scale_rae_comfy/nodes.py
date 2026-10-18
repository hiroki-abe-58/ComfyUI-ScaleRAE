"""ComfyUI nodes for Scale-RAE text-to-image (runs in the ComfyUI process; nothing is downloaded at run time)."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

ScaleRAEModel = io.Custom("SCALE_RAE_MODEL")

PROMPT_HELP = ("The model is a chat model. Requests such as 'Can you generate a photo of ...?' or 'Generate an image of ...' make it start the "
               "image itself (the official path); plain descriptions make it answer with text only (see image_start).")
IMAGE_START_HELP = ("auto: the official path; if the model answers with text only, the image is started right after the prompt instead (same "
                    "seed; the report says so). model: the official path only, a text-only answer is an error. force: always start the image "
                    "right after the prompt.")
GUIDANCE_HELP = ("Classifier-free guidance scale; 1.0 = no guidance (the official default and the setting of the paper's evaluation). "
                 "Above 1.0 see cfg_mode.")
CFG_MODE_HELP = ("standard: guidance > 1 applies classifier-free guidance at every sampling step. official-cli: exactly what the released "
                 "code does, where guidance > 1 has no effect beyond rounding (ZitengWangNYU/Scale-RAE issue #7). Both are identical at guidance 1.0.")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


class ScaleRAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleRAELoader",
            display_name="Scale-RAE Loader",
            category="loaders/scale_rae",
            description=("Loads an official Scale-RAE text-to-image model (Qwen2.5-1.5B language model + 2.4B DiT) and an RAE decoder from "
                         "models/scale_rae. The generator and the decoder must belong to the same encoder space (the WebSSL model with "
                         "webssl300m_decoder). One model stays resident; loading another frees it. No weights are included in this node pack."),
            inputs=[
                io.Combo.Input("generator", options=paths.generators(), tooltip="A nyu-visionx/Scale-RAE-* folder in models/scale_rae"),
                io.Combo.Input("decoder", options=paths.decoders(), tooltip="The matching nyu-visionx/*_decoder folder in models/scale_rae"),
                io.Combo.Input("dtype", options=list(runtime.DTYPES), default="float32",
                               tooltip="float32 is what the official loader uses (about 17.4 GiB peak VRAM; matches the official code). float16 "
                                       "needs about 9.5 GiB and stayed closest to float32 in our tests; bfloat16 also needs about 9.5 GiB but "
                                       "changed the images more. The RAE decoder always runs in float32, and the CPU always runs float32."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM after every run."),
            ],
            outputs=[ScaleRAEModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, generator, decoder, dtype, device, keep_loaded) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.generator_path(generator), paths.decoder_path(decoder), dtype, device, keep_loaded))


class ScaleRAETextToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleRAETextToImage",
            display_name="Scale-RAE Text to Image",
            category="image/scale_rae",
            description=("Text-to-image with Scale-RAE: the language model reads the prompt, its latent queries condition a rectified-flow "
                         "DiT that samples encoder features, and the RAE decoder turns them into pixels. The WebSSL decoder outputs "
                         "224 x 224 images. With batch_size > 1 the seeds are seed, seed + 1, ..."),
            inputs=[
                ScaleRAEModel.Input("model"),
                io.String.Input("prompt", multiline=True, default="Can you create an image of a peaceful garden?", tooltip=PROMPT_HELP),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Same model, prompt, seed, settings and environment give the same image."),
                io.Float.Input("guidance", default=1.0, min=1.0, max=10.0, step=0.1, tooltip=GUIDANCE_HELP),
                io.Combo.Input("cfg_mode", options=list(runtime.CFG_MODES), default="standard", tooltip=CFG_MODE_HELP),
                io.Int.Input("steps", default=50, min=2, max=200, tooltip="Euler steps of the flow sampler (50 = official and paper setting)."),
                io.Int.Input("batch_size", default=1, min=1, max=16),
                io.Combo.Input("image_start", options=list(runtime.IMAGE_STARTS), default="auto", tooltip=IMAGE_START_HELP),
            ],
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, guidance, cfg_mode, steps, batch_size, image_start) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        loaded = runtime.acquire(model)
        info = loaded.info
        images, reports = [], []
        try:
            calls = steps - 1  # DiT evaluations per image (the flow loop + its final Euler step)
            tick = _progress(calls * batch_size)
            mode = image_start
            for i in range(batch_size):
                base = i * calls
                out = runtime.generate(loaded, prompt, int(seed) + i, guidance, steps, cfg_mode=cfg_mode, image_start=mode,
                                       progress=lambda n, base=base: tick(base + n))
                if mode != image_start:  # report what was asked for, and why the image was forced
                    out["report"].update(image_start=image_start, text_only_answer=reports[0]["text_only_answer"])
                images.append(out["image"])
                reports.append(out["report"])
                if mode == "auto" and "text_only_answer" in out["report"]:
                    mode = "force"  # the text part is greedy, so the other seeds would get the same text-only answer: skip it
        finally:
            del loaded  # drop this reference first, otherwise release() cannot free the weights
            if not model.get("keep_loaded", True):
                runtime.release()
        report = runtime.report_json(info, reports[0] if batch_size == 1 else {"images": reports})
        return io.NodeOutput(torch.cat(images, dim=0), report)


class ScaleRAEUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleRAEUnload",
            display_name="Scale-RAE Unload",
            category="loaders/scale_rae",
            description="Frees the resident Scale-RAE model (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect an image so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class ScaleRAEExtension(ComfyExtension):
    async def get_node_list(self):
        return [ScaleRAELoader, ScaleRAETextToImage, ScaleRAEUnload]


async def comfy_entrypoint() -> ScaleRAEExtension:
    paths.register()
    return ScaleRAEExtension()
