set gpu_host=loki.cs.uni-saarland.de
set gpu_user=deme00001
:: Port that jupyter is accessible through in the GPU host machine
set jupyter_docker_port=61372
:: Use this port to access jupyter in your browser
set local_jupyter_port=8888
set docker_script_path=/raid/deme00001/neural-networks-project-vision/docker-t1.sh
ssh -L %local_jupyter_port%:%gpu_host%:%jupyter_docker_port% %gpu_user%@%gpu_host% "%docker_script_path%"