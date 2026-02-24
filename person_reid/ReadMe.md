
## Install & Run 
- step1: conda create -n reid_gui python=3.8
- step2: conda activate reid_gui
- step3: If mac or linux OS: sh install.sh; 
         or win OS:  ./install.bat
- step4: cd GUI; python main.py


## UI modified by Users
- using qt designer
- pyside6-uic home.ui > home.py

## support GPU
> 我们支持GPU去加速推理,目前只支持N卡。需要大家自己配置好cuda/cudnn,然后安装时执行install_gpu.sh or install_gpu.bat