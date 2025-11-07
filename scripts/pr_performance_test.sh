#!/bin/bash
set -e  # Exit immediately if any command fails

# 定义清理函数，用于退出时（包括异常退出）清理资源
cleanup() {
    local exit_code=$?
    echo -e "\n检测到退出，开始清理资源..."
    # 停止SGLang服务器进程（如果存在且运行中）
    if [ -n "$SGLANG_PID" ] && ps -p "$SGLANG_PID" >/dev/null 2>&1; then
        echo "停止SGLang服务器进程（PID: $SGLANG_PID）"
        kill "$SGLANG_PID" >/dev/null 2>&1 || echo "警告：停止服务器进程失败"
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
    # 删除临时日志文件（如果存在）
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "删除临时日志文件: $LOG_FILE"
        rm -f "$LOG_FILE" || echo "警告：删除临时日志文件失败"
    fi
    echo "清理完成。"
    exit $exit_code  # 保持原退出码
}

# 设置陷阱，捕获所有退出信号（正常退出、错误退出、Ctrl+C等）
trap cleanup EXIT

# 检查是否至少传递了2个参数（目标路径和模型路径）
if [ $# -lt 2 ]; then
    echo "错误：请传递至少2个参数！"
    echo "用法：$0 <参数1:目标路径> <参数2:模型路径> [参数3:tp数]"
    exit 1  # 非0退出码表示错误
fi

target_dir="$1"
model_path="$2"
# tp参数默认值为1
tp=${3:-1}
extra_args="$4"

start_sglang() {
    # 创建临时日志文件
    LOG_FILE=$(mktemp /tmp/sglang_server_XXXX.log)
    echo "Server logs will be temporarily stored in: $LOG_FILE"

    # 定义服务器启动参数数组
    args=(
        --model-path "$model_path"
        --tp "$tp"
        --host 127.0.0.1
        --port 8080
        --attention-backend ascend
        # --context-length 8192
        --mem-fraction-static 0.8
        --disable-cuda-graph
        #--enable-metrics
    )

    # 如果存在第4个参数，将其拆分为数组元素添加到参数列表中
    if [ -n "$extra_args" ]; then
        # 将第4个参数按空格拆分为数组，再添加到args中
        read -ra extraargs <<< "$extra_args"
        args+=("${extraargs[@]}")
    fi

    # 启动SGLang服务器（使用参数数组传递所有参数）
    echo "Starting SGLang server..."
    python -m sglang.launch_server "${args[@]}" > "$LOG_FILE" 2>&1 &
    SGLANG_PID=$!  # 记录服务器进程ID
}

wait_sglang_start_success() {
    # 定义服务器启动成功的日志标识
    SUCCESS_LOG="The server is fired up and ready to roll!"
    # 等待服务器启动成功（检查日志关键字）
    echo "Waiting for server startup, monitoring log keyword: '$SUCCESS_LOG'"
    WAIT_TIMEOUT=500  # 超时时间（秒）
    WAIT_INTERVAL=10   # 检查间隔（秒）
    ELAPSED=0
    SUCCESS=0

    while [ $ELAPSED -lt $WAIT_TIMEOUT ]; do
        if grep -q "$SUCCESS_LOG" "$LOG_FILE"; then
            SUCCESS=1
            break
        fi
        echo "Server not ready, waited ${ELAPSED}s (remaining timeout: $((WAIT_TIMEOUT - ELAPSED))s)"
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    done

    # 处理服务器启动结果
    if [ $SUCCESS -eq 1 ]; then
        echo "Detected successful server startup log!"
    else
        echo "Error: Server startup timed out (waited ${WAIT_TIMEOUT}s)"
        echo "===== Last 100 lines of server log ====="
        tail -n 100 "$LOG_FILE"
        echo "==========================="
        exit 1  # 触发cleanup陷阱
    fi
}

prepare_ais_bench_config() {
    # 复制数据集
    echo "Copying dataset..."
    cp -r ~/.cache/modelscope/hub/datasets/gsm8k /opt/benchmark/ais_bench/datasets || {
        echo "错误：复制数据集失败"
        exit 1
    }

    # 修改Python配置文件
    echo "Modifying configuration files..."
    sed -i "s/localhost/127.0.0.1/g" /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py || {
        echo "错误：替换localhost失败"
        exit 1
    }
    sed -i "s|path=\".*\"|path=\"$model_path\"|" /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py || {
        echo "错误：替换path失败"
        exit 1
    }
    sed -i "s|model=\".*\"|model=\"$model_path\"|" /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py || {
        echo "错误：替换model失败"
        exit 1
    }
}

run_ais_bench() {
    sed -i 's/request_rate[[:space:]]*=[[:space:]]*.*/request_rate = '$1',/' /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py || {
        echo "错误：设置 request_rate 失败"
        exit 1
    }
    sed -i 's/batch_size[[:space:]]*=[[:space:]]*.*/batch_size = '$(( $1 * 10 ))',/' /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py || {
        echo "错误：设置batch_size失败"
        exit 1
    }
    cat /opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
    # 执行测试
    echo "Starting ais_bench test execution..."
    ais_bench \
        --models vllm_api_stream_chat \
        --datasets gsm8k_gen_0_shot_cot_str_perf \
        --summarizer default_perf \
        --mode perf \
        --num-prompts $(( $1 * 10 * 5 )) || {
        echo "错误：ais_bench测试执行失败"
        exit 1
    }

    # 移动输出结果
    echo "Moving test results..."
    OUTPUTS_DIR="$(ls -d "outputs/default"/* 2>/dev/null | grep -E '^.*/[0-9]{8}_[0-9]{6}$' | sort | tail -1)/performances/vllm-api-stream-chat"
    if [ ! -d "$OUTPUTS_DIR" ]; then
        echo "错误：未找到有效的输出目录: '$OUTPUTS_DIR'" >&2
        exit 1
    fi
    # 一次性添加两个字段，并保存到原文件
    jq --arg tp_val "$tp" --arg request_rate_val "$1" '. += {"tp": {"total": $tp_val}, "request_rate": {"total": $request_rate_val}}' \
    "$OUTPUTS_DIR/gsm8kdataset.json" > "$OUTPUTS_DIR/gsm8kdataset.json.tmp" && mv "$OUTPUTS_DIR/gsm8kdataset.json.tmp" "$OUTPUTS_DIR/gsm8kdataset.json"
    model=$(basename "$model_path")
    echo "$model"
    mkdir -p "$target_dir/$model"
    mv "$OUTPUTS_DIR" "$target_dir/$model/$1" || {
        echo "错误：移动输出目录失败"
        exit 1
    }
}

main() {
    start_sglang
    wait_sglang_start_success
    prepare_ais_bench_config
    time run_ais_bench 1
    time run_ais_bench 4
    time run_ais_bench 16
}

main

echo "All operations completed successfully!"
