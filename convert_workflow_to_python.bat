@echo off
X:\ComfyUI_windows_portable\python_embeded\python.exe X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\comfyui_to_python.py

if exist comfy_gen.py del /F comfy_gen.py

rename workflow_api.py comfy_gen.py

pause