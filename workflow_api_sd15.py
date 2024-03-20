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


from nodes import (
    KSamplerAdvanced,
    NODE_CLASS_MAPPINGS,
    LoraLoader,
    SaveImage,
    VAEDecode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
        efficient_loader_10 = efficient_loader.efficientloader(
            ckpt_name="analogMadness_v70.safetensors",
            vae_name="Baked VAE",
            clip_skip=-1,
            lora_name="pytorch_lora_weights.safetensors",
            lora_model_strength=1,
            lora_clip_strength=1,
            positive="(best quality masterpiece official art extremely detailed ultra realistic professional photography best lighting) shot on kodak portra 400 film, instagram style, realistic wide angle BREAK photo of (Mckenna Grace:0.8) 18 yo chubby sagging tits (pale skin) (naked:1.3) light pink hair long long hair, captivating poses light smile, on a sailboat, focus on eyes, high heels, feet",
            negative="(worst quality normal quality ugly) photoshop, airbrush, disfigured, kitsch, oversaturated, low-res, Deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb,poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, long neck, long body, disgusting, poorly drawn, mutilated, mangled, conjoined twins, extra legs, extra arms, meme, deformed, elongated, strabismus, heterochromia, watermark, extra fingers, (underwear:1.3), (panties:1.3), asian, chinese, blind eyes, dead eyes small breasts tiny boobs flat chest (old mature) photo frame",
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=512,
            empty_latent_height=512,
            batch_size=1,
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_47 = upscalemodelloader.load_model(
            model_name="4x_NMKD-Superscale-SP_178000_G.pth"
        )

        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_49 = ultralyticsdetectorprovider.doit(
            model_name="bbox/face_yolov8m.pt"
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_50 = samloader.load_model(
            model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
        )

        loraloader = LoraLoader()
        loraloader_52 = loraloader.load_lora(
            lora_name="AdvancedEnhancer.safetensors",
            strength_model=0.65,
            strength_clip=1,
            model=get_value_at_index(efficient_loader_10, 0),
            clip=get_value_at_index(efficient_loader_10, 5),
        )

        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()
        saveimage = SaveImage()

        for q in range(10):
            ksampleradvanced_45 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=6,
                cfg=1.5,
                sampler_name="lcm",
                scheduler="normal",
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(loraloader_52, 0),
                positive=get_value_at_index(efficient_loader_10, 1),
                negative=get_value_at_index(efficient_loader_10, 2),
                latent_image=get_value_at_index(efficient_loader_10, 3),
            )

            vaedecode_46 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_45, 0),
                vae=get_value_at_index(efficient_loader_10, 4),
            )

            ultimatesdupscale_48 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1.5,
                sampler_name="lcm",
                scheduler="normal",
                denoise=0.2,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(vaedecode_46, 0),
                model=get_value_at_index(efficient_loader_10, 0),
                positive=get_value_at_index(efficient_loader_10, 1),
                negative=get_value_at_index(efficient_loader_10, 2),
                vae=get_value_at_index(efficient_loader_10, 4),
                upscale_model=get_value_at_index(upscalemodelloader_47, 0),
            )

            facedetailer_51 = facedetailer.doit(
                guide_size=384,
                guide_size_for=True,
                max_size=1024,
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1.5,
                sampler_name="lcm",
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
                wildcard="best qulity extremely detailed Mckenna Grace loli caucasian (freckles:0.85) symmetrical eyes makeup mascara (thick eyelash) lips (blush) looking at viewer light smile",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=get_value_at_index(ultimatesdupscale_48, 0),
                model=get_value_at_index(efficient_loader_10, 0),
                clip=get_value_at_index(loraloader_52, 1),
                vae=get_value_at_index(efficient_loader_10, 4),
                positive=get_value_at_index(efficient_loader_10, 1),
                negative=get_value_at_index(efficient_loader_10, 2),
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_49, 0),
                sam_model_opt=get_value_at_index(samloader_50, 0),
            )

            saveimage_40 = saveimage.save_images(
                filename_prefix="draft/031724",
                images=get_value_at_index(facedetailer_51, 0),
            )


if __name__ == "__main__":
    main()
