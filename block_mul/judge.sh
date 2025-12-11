#!/bin/bash

# ==========================================================
# 參數配置
# ==========================================================

# 1. 定義要測試的矩陣維度 N 列表
# 請根據您實際生成的檔案清單調整此列表。
TEST_DIMENSIONS=(64 128 256 512 1024 2048) 

# 2. 程式和檔案名稱
DATA_DIR="../dataset"
BLOCK_MUL_EXE="./block_mul_par" # 假設 Makefile 會生成這個名稱的執行檔

# 3. 輸出結果日誌
LOG_FILE="block_mul_test_results.log"

# ==========================================================
# 準備工作
# ==========================================================

echo "--- 矩陣乘法測試腳本 (使用 Makefile) 開始 ---" | tee $LOG_FILE
echo "測試時間: $(date)" | tee -a $LOG_FILE
echo "測試維度: ${TEST_DIMENSIONS[*]}" | tee -a $LOG_FILE
echo "---" | tee -a $LOG_FILE

# 1. 執行 Makefile 的編譯目標 (例如: 'make all' 或 'make judge')
echo "-> 步驟 1: 執行 'make' 進行編譯..." | tee -a $LOG_FILE
make clean
make block_mul_par # 或者使用 'make all'，取決於您的 Makefile
if [ $? -ne 0 ]; then 
    echo "錯誤: Makefile 編譯或 judge 目標失敗！請檢查 Makefile。" | tee -a $LOG_FILE
    exit 1
fi

# 檢查執行檔是否存在
if [ ! -f "$BLOCK_MUL_EXE" ]; then
    echo "錯誤: Makefile 沒有生成預期的執行檔 $BLOCK_MUL_EXE，請檢查 Makefile 的輸出名稱。" | tee -a $LOG_FILE
    exit 1
fi


# ==========================================================
# 循環執行所有測試
# ==========================================================

for N in "${TEST_DIMENSIONS[@]}"; do
    
    echo -e "\n--- 正在測試維度 N=$N ---" | tee -a $LOG_FILE

    A_FILE="${DATA_DIR}/A_${N}"
    B_FILE="${DATA_DIR}/B_${N}"
    C_TRUE_FILE="${DATA_DIR}/C_${N}"

    # 檢查測試數據檔案是否存在
    if [ ! -f "$A_FILE" ] || [ ! -f "$B_FILE" ] || [ ! -f "$C_TRUE_FILE" ]; then
        echo "警告: 缺少 N=$N 的數據檔案，跳過此測試。" | tee -a $LOG_FILE
        continue
    fi
    
    # 2. 執行 Strassen 算法
    echo "-> 步驟 2: 執行 Strassen 算法..." | tee -a $LOG_FILE
    
    # 調用編譯好的執行檔，傳入 N 和三個檔案名稱
    # 執行指令格式: ./strassen_read N A_filename B_filename C_true_filename
    "$BLOCK_MUL_EXE" "$N" "$A_FILE" "$B_FILE" "$C_TRUE_FILE" 2>&1 | tee -a $LOG_FILE
    
    if [ $? -ne 0 ]; then 
        echo "警告: 程式在 N=$N 時執行失敗 (退出碼 $?)。" | tee -a $LOG_FILE
    fi
    
done

# ==========================================================
# 腳本結束
# ==========================================================

echo -e "\n--- 所有測試完成 ---" | tee -a $LOG_FILE
echo "完整結果已保存到 $LOG_FILE"

exit 0