#!/bin/bash
#SBATCH --job-name=create_and_install_env
#SBATCH --partition=64c512g # 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=create_env_%j.log

# 1. 加载 miniconda3 模块
module load miniconda3

# 2. 初始化 Conda，确保后续命令能正确执行
# 这行很重要，它使得 conda 命令在非交互式 shell 中可用
eval "$(conda shell.bash hook)"

# 3. 定义你的 Conda 环境名称
ENV_NAME="U-stats"

# 4. 检查环境是否存在。如果不存在，则创建它。
# -y 选项表示自动同意安装
echo "检查 Conda 环境 '${ENV_NAME}'..."
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo "环境 '${ENV_NAME}' 不存在，正在创建..."
    conda create -y -n "${ENV_NAME}" python=3.11 # 建议指定一个Python版本
    echo "环境 '${ENV_NAME}' 创建成功。"
else
    echo "环境 '${ENV_NAME}' 已存在。"
fi

# 5. 激活 Conda 环境
conda activate "${ENV_NAME}"
echo "已激活环境: $(conda env list | grep '*' | awk '{print $1}')"


# 6. 安装或更新所需的包
# 你可以根据你的 requirements.txt 或者直接列出包
echo "开始安装/更新包..."


# 其他包建议用 pip 安装，除非 conda 源有更好的版本或特殊需求
python -m pip install --upgrade u-stats

pip install torch pandas matplotlib networkx seaborn scipy igraph psutil

echo "所有包安装/更新完成！"

# 可选：显示已安装的包列表，方便调试
pip list

echo "Slurm 作业完成。"