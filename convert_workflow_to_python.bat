@echo off
X:\ComfyUI_windows_portable\python_embeded\python.exe X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\comfyui_to_python.py

:: Rename the output file
rename workflow_api.py comfy_gen.py

pause