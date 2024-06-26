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
    VAEDecode,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    CLIPSetLastLayer,
    SaveImage,
    NODE_CLASS_MAPPINGS,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_110 = checkpointloadersimple.load_checkpoint(
            ckpt_name="aiponyanime_v1.safetensors"
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_111 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2,
            clip=get_value_at_index(checkpointloadersimple_110, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_112 = cliptextencode.encode(
            text="1girl, ginger red head green eyes leaning over, (short sweater dress) off shoulder, cleavage, sexy, seductive, long hair, lips, small waist looking at the viewer revealing bokeh (light particles choker)",
            clip=get_value_at_index(clipsetlastlayer_111, 0),
        )

        cliptextencode_113 = cliptextencode.encode(
            text="(bad quality:1.4), unaestheticXL_Alb2 signature watermark logo signature, (censored) (bra) (belt)",
            clip=get_value_at_index(clipsetlastlayer_111, 0),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_114 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(10):
            freeu_v2_105 = freeu_v2.patch(
                b1=1.1,
                b2=1.1500000000000001,
                s1=0.85,
                s2=0.35000000000000003,
                model=get_value_at_index(checkpointloadersimple_110, 0),
            )

            ksampleradvanced_99 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=25,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(freeu_v2_105, 0),
                positive=get_value_at_index(cliptextencode_112, 0),
                negative=get_value_at_index(cliptextencode_113, 0),
                latent_image=get_value_at_index(emptylatentimage_114, 0),
            )

            vaedecode_101 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_99, 0),
                vae=get_value_at_index(checkpointloadersimple_110, 2),
            )

            saveimage_71 = saveimage.save_images(
                filename_prefix="draft/042824",
                images=get_value_at_index(vaedecode_101, 0),
            )


if __name__ == "__main__":
    main()
