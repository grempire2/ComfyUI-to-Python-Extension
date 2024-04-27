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


from nodes import KSamplerAdvanced, NODE_CLASS_MAPPINGS, VAEDecode, SaveImage


def main():
    import_custom_nodes()
    with torch.inference_mode():
        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
        efficient_loader_48 = efficient_loader.efficientloader(
            ckpt_name="aiponyanime_v1.safetensors",
            vae_name="Baked VAE",
            clip_skip=-2,
            lora_name="None",
            lora_model_strength=1,
            lora_clip_strength=1,
            positive="1girl robot girl, (robot joints, intervention unit, glowing chest), full body, white mechanical head, bangs, arched back, hoodie, covered nipples, provocative, cosmic, cables electrical wires, smirk,covered face",
            negative="(bad quality:1.4), unaestheticXL_Alb2 signature watermark logo signature, (censored), (navel)",
            token_normalization="mean",
            weight_interpretation="A1111",
            empty_latent_width=1024,
            empty_latent_height=1024,
            batch_size=1,
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
                model=get_value_at_index(efficient_loader_48, 0),
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
                positive=get_value_at_index(efficient_loader_48, 1),
                negative=get_value_at_index(efficient_loader_48, 2),
                latent_image=get_value_at_index(efficient_loader_48, 3),
            )

            vaedecode_101 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_99, 0),
                vae=get_value_at_index(efficient_loader_48, 4),
            )

            saveimage_87 = saveimage.save_images(
                filename_prefix="042524", images=get_value_at_index(vaedecode_101, 0)
            )


if __name__ == "__main__":
    main()
