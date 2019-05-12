@echo off
call env\Scripts\activate.bat
echo [VirtualEnv Activated]
explorer "http://localhost:8097"
visdom -logging_level WARN
