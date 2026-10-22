# #!/bin/bash

# # 定义结果文件的根目录
# RESULTS_ROOT="/data1/wzy/sub-learn/results/record"
# # 你的 eval_math.py 脚本路径
# SCRIPT_PATH="/data1/wzy/sub-learn/src/eval_math.py"

# echo "Starting batch re-evaluation for all specified directories..."

# # -----------------------------------------------------------------
# # 1. 处理 'base_vector' 和 'gt_base_vector' 目录
# # -----------------------------------------------------------------
# # for DIR_NAME in "base_vector" "gt_base_vector"
# for DIR_NAME in "math_base_vector"
# do
#     echo "Processing directory: ${DIR_NAME}..."
#     for i in $(seq 0 31)
#     do
#         INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/use_base_vector_layers_${i}_direct_q_1.0.json"
        
#         if [ -f "${INPUT_FILE}" ]; then
#             echo "  - Running on layer $i..."
#             # 运行单独的命令，每次只处理一个文件，绕过Python脚本的循环
#             # python "${SCRIPT_PATH}" --mode accuracy --result_file "${INPUT_FILE}"
#             python "${SCRIPT_PATH}" result_file="${INPUT_FILE}"
#         else
#             echo "  - Warning: File not found: ${INPUT_FILE}"
#         fi
#     done
# done
# echo "--------------------------------------------------"

# # -----------------------------------------------------------------
# # 2. 处理 'licv_version' 和 'gt_licv_version' 目录
# # -----------------------------------------------------------------
# # for DIR_NAME in "licv_version" "gt_licv_version"
# for DIR_NAME in "math_licv_version"
# do
#     echo "Processing directory: ${DIR_NAME}..."
#     for i in $(seq 0 31)
#     do
#         INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/licv_layers_${i}_direct_q_1.0.json"
        
#         if [ -f "${INPUT_FILE}" ]; then
#             echo "  - Running on layer $i..."
#             # python "${SCRIPT_PATH}" --mode accuracy --result_file "${INPUT_FILE}"
#             python "${SCRIPT_PATH}" result_file="${INPUT_FILE}"
#         else
#             echo "  - Warning: File not found: ${INPUT_FILE}"
#         fi
#     done
# done
# echo "--------------------------------------------------"

# # -----------------------------------------------------------------
# # 3. 处理 'mimic_vector' 和 'gt_mimic_vector' 目录
# # -----------------------------------------------------------------
# # for DIR_NAME in "mimic_vector" "gt_mimic_vector"
# for DIR_NAME in "math_mimic_vector" 
# do
#     echo "Processing directory: ${DIR_NAME}..."
#     for i in $(seq 0 31)
#     do
#         INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/mimic_layers_${i}_direct_q_1.0.json"
        
#         if [ -f "${INPUT_FILE}" ]; then
#             echo "  - Running on layer $i..."
#             # python "${SCRIPT_PATH}" --mode accuracy --result_file "${INPUT_FILE}"
#             python "${SCRIPT_PATH}" result_file="${INPUT_FILE}"
#         else
#             echo "  - Warning: File not found: ${INPUT_FILE}"
#         fi
#     done
# done
# echo "--------------------------------------------------"

# # -----------------------------------------------------------------
# # 4. 处理 'icl-llama-3.1-8b-instruct-gsm8k-0shot-direct' (单个文件)
# # -----------------------------------------------------------------
# # DIR_NAME="icl-llama-3.1-8b-instruct-gsm8k-0shot-direct"
# # INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/0shot.json"

# # if [ -f "${INPUT_FILE}" ]; then
# #     echo "Processing single file: ${DIR_NAME}..."
# #     python "${SCRIPT_PATH}" --mode accuracy --result_file "${INPUT_FILE}"
# # else
# #     echo "Warning: File not found: ${INPUT_FILE}"
# # fi
# # echo "--------------------------------------------------"

# echo "All re-evaluation tasks completed."

#!/bin/bash

# 定义结果文件的根目录
RESULTS_ROOT="/data1/wzy/sub-learn/results/record"
# 你的 eval_math.py 脚本路径
SCRIPT_PATH="/data1/wzy/sub-learn/src/eval_math.py"

echo "Starting batch re-evaluation for all specified directories..."

# -----------------------------------------------------------------
# 1. 处理 'base_vector' 和 'gt_base_vector' 目录
# -----------------------------------------------------------------
for DIR_NAME in "math_base_vector"
do
    echo "Processing directory: ${DIR_NAME}..."
    for i in $(seq 0 31)
    do
        INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/use_base_vector_layers_${i}_direct_q_1.0.json"
        
        if [ -f "${INPUT_FILE}" ]; then
            echo "  - Running on layer $i..."
            # 使用 +result_file 来追加配置
            python "${SCRIPT_PATH}" +result_file="${INPUT_FILE}"
        else
            echo "  - Warning: File not found: ${INPUT_FILE}"
        fi
    done
done
echo "--------------------------------------------------"

# -----------------------------------------------------------------
# 2. 处理 'licv_version' 和 'gt_licv_version' 目录
# -----------------------------------------------------------------
for DIR_NAME in "math_licv_version"
do
    echo "Processing directory: ${DIR_NAME}..."
    for i in $(seq 0 31)
    do
        INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/licv_layers_${i}_direct_q_1.0.json"
        
        if [ -f "${INPUT_FILE}" ]; then
            echo "  - Running on layer $i..."
            python "${SCRIPT_PATH}" +result_file="${INPUT_FILE}"
        else
            echo "  - Warning: File not found: ${INPUT_FILE}"
        fi
    done
done
echo "--------------------------------------------------"

# -----------------------------------------------------------------
# 3. 处理 'mimic_vector' 和 'gt_mimic_vector' 目录
# -----------------------------------------------------------------
for DIR_NAME in "math_mimic_vector" 
do
    echo "Processing directory: ${DIR_NAME}..."
    for i in $(seq 0 31)
    do
        INPUT_FILE="${RESULTS_ROOT}/${DIR_NAME}/mimic_layers_${i}_direct_q_1.0.json"
        
        if [ -f "${INPUT_FILE}" ]; then
            echo "  - Running on layer $i..."
            python "${SCRIPT_PATH}" +result_file="${INPUT_FILE}"
        else
            echo "  - Warning: File not found: ${INPUT_FILE}"
        fi
    done
done
echo "--------------------------------------------------"

echo "All re-evaluation tasks completed."