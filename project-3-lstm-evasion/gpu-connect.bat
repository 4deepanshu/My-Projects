set gpu_host=thor.cs.uni-saarland.de
set gpu_user=team2
:: Port that jupyter is accessible through in the GPU host machine
set jupyter_docker_port=61242
:: Use this port to access jupyter in your browser
set local_jupyter_port=8888
set docker_script_path=/raid/mlcysec20/TheHiddenAdversaries/git/docker.sh
ssh -L %local_jupyter_port%:%gpu_host%:%jupyter_docker_port% %gpu_user%@%gpu_host% "%docker_script_path%"