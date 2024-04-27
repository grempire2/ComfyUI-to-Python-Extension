@echo off

set "file=X:\ComfyUI_windows_portable\ComfyUI\ComfyUI-to-Python-Extension\workflow_api.py"
set "tempFile=temp.py"

:: Create a new temporary file
powershell -Command "$content = Get-Content %file%; $newContent = @(); foreach ($line in $content) { if ($line -match '\bnegative=\".*\",') { $newLine = $line -replace '\bnegative=\".*\",', 'negative=negative_,'; $newContent += $newLine } elseif ($line -match '\bpositive=\".*\",') { $newLine = $line -replace '\bpositive=\".*\",', 'positive=positive_,'; $newContent += $newLine } elseif ($line -match '\bckpt_name=\".*\",') { $newLine = $line -replace '\bckpt_name=\".*\",', 'ckpt_name=ckpt_name_,'; $newContent += $newLine } elseif ($line -match '\bempty_latent_width=\d+,') { $newLine = $line -replace '\bempty_latent_width=\d+,', 'empty_latent_width=empty_latent_width_,'; $newContent += $newLine } elseif ($line -match '\bempty_latent_height=\d+,') { $newLine = $line -replace '\bempty_latent_height=\d+,', 'empty_latent_height=empty_latent_height_,'; $newContent += $newLine } else { $newContent += $line } }; $newContent | Set-Content %tempFile%"

:: Add the new variables and argument parsing at the very top of the script
powershell -Command "$content = Get-Content %tempFile%; 'import argparse', 'parser = argparse.ArgumentParser()', 'parser.add_argument(\"--positive_\", type=str, required=True)', 'parser.add_argument(\"--negative_\", type=str, required=True)', 'parser.add_argument(\"--ckpt_name_\", type=str, required=True)', 'parser.add_argument(\"--empty_latent_width_\", type=int, required=True)', 'parser.add_argument(\"--empty_latent_height_\", type=int, required=True)', 'args = parser.parse_args()', 'positive_ = args.positive_', 'negative_ = args.negative_', 'ckpt_name_ = args.ckpt_name_', 'empty_latent_width_ = args.empty_latent_width_', 'empty_latent_height_ = args.empty_latent_height_', $content | Set-Content %tempFile%"

:: Replace the original file with the modified one
move /Y %tempFile% %file%

pause