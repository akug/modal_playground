
## Setup ControlNet-XS on WSL

```bash
conda create -n xs python=3.9 # 10 and 11 didn't work, because tokenizer couldn't be built
conda activate xs
# install dependencies
sudo apt install pkg-config build-essential
# install rust; Check here for up-to-date installation instructions
#  https://www.rust-lang.org/tools/install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# get the repo, install according to their README.md
git clone https://github.com/vislearn/ControlNet-XS.git
pip install -r requirements/pt2.txt
# install dependencies for SD-XL
pip install "diffusers>0.21.4" invisible_watermark "transformers>=4.34.1" accelerate safetensors

```