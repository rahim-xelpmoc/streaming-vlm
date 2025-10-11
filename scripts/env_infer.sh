conda create -n streamingvlm-infer python=3.11 -y
conda activate streamingvlm-infer
conda install -y ffmpeg -y

pip install -r infer_requirements.txt
pip install transformers==4.51.3 accelerate deepspeed peft opencv-python decord datasets tensorboard gradio pillow-heif gpustat timm sentencepiece openai av==12.0.0 liger_kernel numpy==1.24.4 yt-dlp tqdm huggingface_hub ffmpeg wandb
pip install torch==2.7.1 torchvision torchaudio==2.7.1 qwen_vl_utils==0.0.11 
pip install -e streaming_vlm/livecc_utils/
# install flash-attn
# find your version at https://github.com/Dao-AILab/flash-attention/releases
# example:
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
