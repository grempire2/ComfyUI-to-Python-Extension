@echo off
X:\ComfyUI_windows_portable\python_embeded\python.exe X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\comfyui_to_python.py

set "file=X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\workflow_api.py"
set "tempFile=temp.py"

:: Create a new temporary file
powershell -Command "$content = Get-Content %file%; $newContent = @(); foreach ($line in $content) { if ($line -match '\bnegative=\".*\",') { $newLine = $line -replace '\bnegative=\".*\",', 'negative=negative_,'; $newContent += $newLine } else { $newContent += $line } }; $newContent | Set-Content %tempFile%"

:: Add the new variable at the top of the script after the imports
powershell -Command "$content = Get-Content %tempFile%; $content[0], 'negative_ = \"\"', $content[1..($content.Length - 1)] | Set-Content %tempFile%"

:: Replace the original file with the modified one
move /Y %tempFile% %file%

pause