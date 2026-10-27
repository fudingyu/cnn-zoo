@echo off
chcp 65001 >nul
cd /d %~dp0

rem Point RYZEN_AI_PYTHON at the interpreter that has AMD's onnxruntime
rem (VitisAIExecutionProvider). Defaults to whatever is on PATH.
if not defined RYZEN_AI_PYTHON set RYZEN_AI_PYTHON=python

echo ============================================================
echo   CNN Zoo Studio - NPU / CPU / iGPU 三方实测台
echo   python: %RYZEN_AI_PYTHON%
echo ============================================================
echo.

"%RYZEN_AI_PYTHON%" zoo_server.py --port 8780
pause
