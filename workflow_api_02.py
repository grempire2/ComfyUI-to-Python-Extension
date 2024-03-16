import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

_positive = "(best quality materpiece official art cinematic bokeh) anime style\nBREAK\n(loli:0.9) korean bombshell (rainbow hair), large breasts cleavage, detailed eyes green eyes\nBREAK\nPlayful romper in a vibrant pattern, worn with platform sandals, tassel earrings (bubbles)"
_negative = "worst quality, ugly, western comics, haze, muted colors, deformed, bad eyes musculine elf naked nude nipples"


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import SaveImage, ImageScaleBy, LoadImage, NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
        efficient_loader_54 = efficient_loader.efficientloader(
            ckpt_name="envysunflowerxl01turbo_v10.safetensors",
            vae_name="Baked VAE",
            clip_skip=-1,
            lora_name="sdxl_lightning_4step_lora.safetensors",
            lora_model_strength=1,
            lora_clip_strength=1,
            positive=_positive,
            negative=_negative,
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=1024,
            empty_latent_height=1024,
            batch_size=1,
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_56 = upscalemodelloader.load_model(
            model_name="4x_NMKD-Superscale-SP_178000_G.pth"
        )

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_65 = ultralyticsdetectorprovider.doit(
            model_name="bbox/face_yolov8m.pt"
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_72 = samloader.load_model(
            model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
        )

        loadimage = LoadImage()
        loadimage_83 = loadimage.load_image(image="074353mjlov8on0azmogo2.jpg")

        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        facedetailerpipe = NODE_CLASS_MAPPINGS["FaceDetailerPipe"]()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        saveimage = SaveImage()
        imagescaleby = ImageScaleBy()

        for q in range(10):
            facedetailer_60 = facedetailer.doit(
                guide_size=384,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=37,
                cfg=31.200000000000003,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7000000000000001,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=get_value_at_index(loadimage_83, 0),
                model=get_value_at_index(efficient_loader_54, 0),
                clip=get_value_at_index(efficient_loader_54, 5),
                vae=get_value_at_index(efficient_loader_54, 4),
                positive=get_value_at_index(efficient_loader_54, 1),
                negative=get_value_at_index(efficient_loader_54, 2),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_65, 0),
                sam_model_opt=get_value_at_index(samloader_72, 0),
            )

            facedetailerpipe_68 = facedetailerpipe.doit(
                guide_size=384,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.5,
                feather=5,
                noise_mask=True,
                force_inpaint=False,
                bbox_threshold=0.5,
                bbox_dilation=10,
                bbox_crop_factor=3,
                sam_detection_hint="center-1",
                sam_dilation=0,
                sam_threshold=0.93,
                sam_bbox_expansion=0,
                sam_mask_hint_threshold=0.7000000000000001,
                sam_mask_hint_use_negative="False",
                drop_size=10,
                refiner_ratio=0.2,
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=get_value_at_index(facedetailer_60, 0),
                detailer_pipe=get_value_at_index(facedetailer_60, 4),
            )

            ultimatesdupscale_52 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1,
                sampler_name="euler_ancestral",
                scheduler="sgm_uniform",
                denoise=0.19,
                mode_type="Linear",
                tile_width=1024,
                tile_height=1024,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(facedetailerpipe_68, 0),
                model=get_value_at_index(efficient_loader_54, 0),
                positive=get_value_at_index(efficient_loader_54, 1),
                negative=get_value_at_index(efficient_loader_54, 2),
                vae=get_value_at_index(efficient_loader_54, 4),
                upscale_model=get_value_at_index(upscalemodelloader_56, 0),
            )

            saveimage_59 = saveimage.save_images(
                filename_prefix="031624",
                images=get_value_at_index(ultimatesdupscale_52, 0),
            )

            imagescaleby_74 = imagescaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=2,
                image=get_value_at_index(facedetailerpipe_68, 0),
            )

            saveimage_82 = saveimage.save_images(
                filename_prefix="031624", images=get_value_at_index(imagescaleby_74, 0)
            )


if __name__ == "__main__":
    main()
