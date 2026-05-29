# SGLang Ascend

## Quick Reference

- [SGLang Documentation](https://docs.sglang.io/)
- [GitHub Repository](https://github.com/sgl-project/sglang)

## SGLang

SGLang is a high-performance serving framework for large language models and multimodal models. It is designed to
deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed
clusters.

---

## Supported Tags and Dockerfile Source

Tags follow this pattern:

```
<sglang_version>-<cann_version>-<chip_series>
```

| Field                 | Example Values            | Description           |
|-----------------------|---------------------------|-----------------------|
| `<sglang_version>`    | `main`, `v0.5.12`         | The version of SGLang |
| `<cann_version>`      | `cann8.5.0`, `cann9.0.0`  | The version of CANN   |
| `<chip_series>`       | `a3`, `910b`              | Target Ascend chip series |

[Dockerfile Source](https://github.com/sgl-project/sglang/blob/main/docker/npu.Dockerfile)

---

## Quick Start

### Prerequisites

#### Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. If the driver is
already installed, skip this section.

See the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=Ubuntu&InstallType=local)
for detailed instructions.

### Run Container

```bash
docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
--net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci8 \
--device=/dev/davinci9 \
--device=/dev/davinci10 \
--device=/dev/davinci11 \
--device=/dev/davinci12 \
--device=/dev/davinci13 \
--device=/dev/davinci14 \
--device=/dev/davinci15 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--entrypoint=bash \
quay.io/ascend/sglang:${TAG}
```

__Note__

- Replace `${TAG}` with the desired tag, e.g., `v0.5.12-cann8.5.0-a3`.
- `davinci8` through `davinci15` should be removed for the 910b chip series.
- If the model is stored locally, mount the local model directory into the container (e.g., `-v /path/to/models:/models`).

### How to Build

```bash
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
# If there are network errors, please modify the Dockerfile to use offline dependencies or use a proxy
# <arch_tag> is the target architecture of the image, e.g., amd64, arm64
docker build --build-arg TARGETARCH=<arch_tag> -t <image_name> -f npu.Dockerfile .
```

### Development

```bash
# Use the SGLang NPU image as base
FROM quay.io/ascend/sglang:${TAG}

# Install required software
RUN apt-get update -y && \
    apt-get install -y <required_software>
...
```

## Supported Hardware

| Chip Series   | Product Examples  | Architecture  |
|---------------|-------------------|---------------|
| a3            | Atlas 800I A3     | arm64, x86_64 |
| 910b          | Atlas 800I A2     | arm64, x86_64 |

## License

View the [LICENSE information](https://github.com/sgl-project/sglang/blob/main/LICENSE) for more details.

As with all container images, pre-installed packages (e.g., OS, runtime, tools, and libraries) may be subject to
their own licenses.
