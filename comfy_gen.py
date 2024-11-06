import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    SaveImage,
    CLIPSetLastLayer,
    CLIPTextEncode,
    LoraLoaderModelOnly,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    CheckpointLoaderSimple,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="aiponyanime_v2.safetensors"
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_10 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="1girl, young teen, (pale), one piece nami,\n \nslim waist, sexy, revealing, skimpy, seductive, covered nipples,\n\n(bride:1.4), lace, earrings, pubic hair, sheer skirt, stockings,\ncolorful, bouquet, ribbons, bracelet,\n\nlooking at viewer,\n\n(surprised), open mouth smile, sweat, constricted pupils, spoken exclamation, sweat,\n\n(blank background, white background),",
            clip=get_value_at_index(clipsetlastlayer_10, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="(bad quality:1.4),(worst quality:1.4),unaestheticXL_Alb2,guly,(censored),monochrome,blurry, lowres,watermark,(old),(mature),asian,(nsfw),nipples,skinny,phone,bed",
            clip=get_value_at_index(clipsetlastlayer_10, 0),
        )

        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        freeu_v2_25 = freeu_v2.patch(
            b1=1.1,
            b2=1.1500000000000001,
            s1=0.85,
            s2=0.35000000000000003,
            model=get_value_at_index(checkpointloadersimple_4, 0),
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_48 = loraloadermodelonly.load_lora_model_only(
            lora_name="1llum1XLP.safetensors",
            strength_model=0.65,
            model=get_value_at_index(freeu_v2_25, 0),
        )

        loraloadermodelonly_49 = loraloadermodelonly.load_lora_model_only(
            lora_name="yandere_trance_v1_pruned.safetensors",
            strength_model=0.65,
            model=get_value_at_index(loraloadermodelonly_48, 0),
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_55 = upscalemodelloader.load_model(
            model_name="4x-AnimeSharp.pth"
        )

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_59 = ultralyticsdetectorprovider.doit(
            model_name="bbox/Eyeful_v2-Paired.pt"
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_60 = samloader.load_model(
            model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
        )

        loadimage = LoadImage()
        loadimage_80 = loadimage.load_image(
            image="cow girl code geass euphemia princess pov_00001_.png"
        )

        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        saveimage = SaveImage()

        for q in range(10):
            ultimatesdupscale_51 = ultimatesdupscale.upscale(
                upscale_by=4,
                seed=random.randint(1, 2**64),
                steps=25,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=0.1,
                mode_type="Linear",
                tile_width=832,
                tile_height=1216,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(loadimage_80, 0),
                model=get_value_at_index(loraloadermodelonly_49, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
                upscale_model=get_value_at_index(upscalemodelloader_55, 0),
            )

            facedetailer_63 = facedetailer.doit(
                guide_size=512,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=25,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=0.17,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                bbox_threshold=0.2,
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
                image=get_value_at_index(ultimatesdupscale_51, 0),
                model=get_value_at_index(loraloadermodelonly_49, 0),
                clip=get_value_at_index(clipsetlastlayer_10, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_59, 0),
                sam_model_opt=get_value_at_index(samloader_60, 0),
            )

            saveimage_57 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(facedetailer_63, 0)
            )


if __name__ == "__main__":
    main()
