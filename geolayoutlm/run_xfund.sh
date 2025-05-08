#!/bin/bash

# 定义所有语言的简称（根据实际数据集调整）
LANGS=("ja" "fr" "de" "es" "it" "pt" "zh")

# 循环处理每种语言
for lang in "${LANGS[@]}"; do
    # 创建 tmux 会话并启动训练任务
    tmux new-session -d -s "xfund_${lang}" "CUDA_VISIBLE_DEVICES=0 python train.py --config=configs/finetune_xfund_${lang}.yaml 2>&1 | tee logs/xfund_${lang}.log"
    
    echo "已启动 ${lang} 的训练任务（会话名：xfund_${lang}）"
    
    # 等待当前会话任务完成
    while tmux has-session -t "xfund_${lang}" 2>/dev/null; do
        sleep 60  # 每分钟检查一次会话状态
    done
    
    echo "${lang} 的训练任务已完成"
done

echo "所有语言训练任务执行完毕！"
