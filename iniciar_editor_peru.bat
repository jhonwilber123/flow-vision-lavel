@echo off
REM Starts the Peru AABB vehicle label editor.
cd /d "%~dp0"
start "" http://127.0.0.1:8877
python server.py
pause
