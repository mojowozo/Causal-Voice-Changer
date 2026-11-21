@echo off
rem Opens the training app with the PyTorch build for older NVIDIA cards: GeForce GTX 900/10 series (a GTX 1080,
rem say), TITAN X/Xp/V and Quadro M/P. The usual build (CUDA 13) no longer runs on them; this one is built
rem for CUDA 12.6. run-training.bat picks it by itself for these cards: use this one if it did not.
setlocal
set "CAUSALVC_GPU=cuda-pascal"
call "%~dp0run-training.bat" %*
