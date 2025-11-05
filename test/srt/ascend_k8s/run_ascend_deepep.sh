pip install kubernetes
pip3 install xgrammar==0.1.25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_goveror
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

export SGLANG_SET_CPU_AFFINITY=1

export PYTHONPATH=/data/.cache/sglang/python:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export GLOO_SOCKET_IFNAME=enp23s0f3
export HCCL_SOCKET_IFNAME=enp23s0f3

#export HCCL_BUFFSIZE=600
#export TASK_QUEUE_ENABLE=2
#export ENBALE_PROFILING=0
#export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=48

cd /data/.cache/sglang/test/srt/ascend_k8s/
# python3 -u test_ascend_disaggregation_deepep_k8s.py
python3 -u test_ascend_disaggregation_deepep_k8s_2p1d.py
# python3 -u test_ascend_disaggregation_deepep_k8s_1p1d.py
