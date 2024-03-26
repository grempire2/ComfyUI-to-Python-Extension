@echo off
X:\ComfyUI_windows_portable\python_embeded\python.exe X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\comfyui_to_python.py
powershell -Command "(Get-Content X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\workflow_api.py) -replace 'for q in range\(10\):', 'for q in range(3):' | Set-Content X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\workflow_api.py"
pause