# System Deployment Dependencies

This document provides a comprehensive list of all system, hardware, and library dependencies required to run the `droidslam_offline_processing.sh` script or DROID-SLAM inference on a new computer with the same setup.

## 1. Hardware & System Requirements
* **GPU**: NVIDIA GPU with at least 11 GB of VRAM. (The current machine uses a **Quadro RTX 8000**).
* **OS**: Linux (Ubuntu 20.04/22.04 recommended).
* **NVIDIA Drivers**: Proprietary drivers must be installed.
  * **Current Machine Exact Driver Version**: `580.126.09`
  * **Current Machine CUDA Version**: `13.0`
* **CUDA Toolkit**: Required for compiling custom C++ and CUDA extensions (`droid-backends` and `lietorch`). 
* **Build Essentials**: Standard C++ compilers (`gcc`, `g++`, `make`).

### System Installation Commands (Ubuntu/Debian)
```bash
# Update package lists and install basic build tools and Python prerequisites
sudo apt-get update
sudo apt-get install -y build-essential python3 python3-pip python3-venv git

# Install the exact NVIDIA driver (Version 580)
sudo apt-get install -y nvidia-driver-580

# It is highly recommended to reboot your machine after installing/updating NVIDIA drivers
sudo reboot
```

## 2. Docker (Optional but Recommended)
If deploying via Docker instead of natively directly on the host machine, you will need the NVIDIA Container Toolkit.

### Docker Installation Commands
```bash
# 1. Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Add the package repositories and install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker runtime and restart Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Python Dependencies
The system relies on a Python 3.10 environment, typically isolated using a Python `venv` or Conda environment.

### Python Environment Installation Commands
```bash
# 1. Clone the repository (if not already copied over)
# Make sure to include submodules!
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git ~/DROID-SLAM
cd ~/DROID-SLAM
# If already cloned but missing submodules, run: git submodule update --init --recursive

# 2. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade basic Python packaging tools
pip install --upgrade pip wheel

# 4. Install PyTorch with CUDA support
# (Match the CUDA version installed on the new machine. Example below is standard)
pip install torch torchvision torchaudio

# 5. Install standard pip requirements required by DROID-SLAM
pip install opencv-python open3d scipy evo tqdm gdown tensorboard pyyaml

# 6. Install custom compiled third-party C++/CUDA extensions 
pip install thirdparty/pytorch_scatter
pip install thirdparty/lietorch

# 7. Finally, compile and install the DROID-SLAM backend
pip install -e .
```

## 4. Hardcoded Paths to Update
When moving `droidslam_offline_processing.sh` to a different machine, be aware of the following hard-coded paths. **You must change these inside the `.sh` script** so they match your new system's directory structure:

* **Virtual Environment Path**: 
  * Current: `~/DROID-SLAM/.venv/bin/activate`
  * Action: Update to the location where you initialized your new `venv`.
* **Input Path Dataset**: 
  * Current: `/home/mbo/bigfoot-FoMo/ijrr`
  * Action: Change to where the input datasets are stored on the new machine.
* **Output Path**: 
  * Current: `/home/mbo/hdd/droidslam-offline/droidslam-${trajectory}`
  * Action: Change to the appropriate output drive/folder.
* **Model Weights Path**: 
  * Current: `/home/mbo/legs_ws/src/droid_slam_ros/droid.pth`
  * Action: Download `droid.pth` on the new machine and update this absolute path.
