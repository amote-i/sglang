#!/bin/bash
set -euo pipefail

PIP_INSTALL="pip install --no-cache-dir"
DEVICE_TYPE=$1


# Install the required dependencies in CI.
apt update -y && apt install -y \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    ca-certificates \
    jq
update-ca-certificates
python3 -m ${PIP_INSTALL} --upgrade pip

### Download MemFabricV2
pip install mf-adapter==1.0.0


### Install vLLM
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG
(cd vllm && VLLM_TARGET_DEVICE="empty" ${PIP_INSTALL} -v -e .)


### Install PyTorch and PTA
PYTORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
${PIP_INSTALL} torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION

PTA_VERSION="v7.2.0-pytorch2.8.0"
PTA_NAME="torch_npu-2.8.0.post2+git95d6260-cp311-cp311-linux_aarch64.whl"
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl"
wget -O "${PTA_NAME}" "${PTA_URL}" && ${PIP_INSTALL} "./${PTA_NAME}"

# Update pip
pip config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple/"
pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn"
pip install --upgrade pip

# ### Install Triton-Ascend
${PIP_INSTALL} attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
pip install triton-ascend==3.2.0rc3

### Install BiSheng
BISHENG_NAME="Ascend-BiSheng-toolkit_aarch64.run"
BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${BISHENG_NAME}"
wget -O "${BISHENG_NAME}" "${BISHENG_URL}" && chmod a+x "${BISHENG_NAME}" && "./${BISHENG_NAME}" --install && rm "${BISHENG_NAME}"


### Install sgl-kernel-npu
SGL_KERNEL_NPU_TAG="20251106"
git clone --depth 1 https://github.com/sgl-project/sgl-kernel-npu.git --branch ${SGL_KERNEL_NPU_TAG}
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
pip install wheel==0.45.1
(cd sgl-kernel-npu && bash ./build.sh && pip install output/deep_ep*.whl output/sgl_kernel_npu*.whl && cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so)


### Install CustomOps (TODO: to be removed once merged into sgl-kernel-npu)
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl

### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[srt_npu]"
