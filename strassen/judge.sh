# ...existing code...
#!/bin/bash

# Usage:
#   ./judge.sh [seq|par|both]
#   or set env vars: SRUN, NVPROF, MODE
# Example:
#   MODE=par NVPROF=nvprof ./judge.sh par
#   SRUN="" ./judge.sh par    # disable srun prefix

MODE_ARG="$1"
MODE="${MODE_ARG:-${MODE:-both}}"   # seq, par, or both (default both)

# SRUN prefix for GPU runs (can be overridden outside)
SRUN=${SRUN:-"srun -p nvidia -n1 --gres=gpu:1"}
# NVPROF or other profiler prefix (empty by default)
NVPROF=${NVPROF:-""}

# ==========================================================
# 參數配置
# ==========================================================
TEST_DIMENSIONS=(64 128 256 512 1024 2048 4096)

DATA_DIR="../dataset"
STRASSEN_PAR_EXE="./strassen_par"
STRASSEN_SEQ_EXE="./strassen_seq"

LOG_FILE="strassen_test_results.log"

echo "--- 矩陣乘法測試腳本 ---" | tee $LOG_FILE
echo "測試時間: $(date)" | tee -a $LOG_FILE
echo "執行模式: $MODE" | tee -a $LOG_FILE
echo "測試維度: ${TEST_DIMENSIONS[*]}" | tee -a $LOG_FILE
echo "---" | tee -a $LOG_FILE

# 編譯（同時編譯 seq 與 par）
echo "-> 編譯程式..." | tee -a $LOG_FILE
make clean
make strassen_par strassen_seq
if [ $? -ne 0 ]; then
    echo "錯誤: 編譯失敗" | tee -a $LOG_FILE
    exit 1
fi

# 檢查執行檔
if [[ "$MODE" == "seq" || "$MODE" == "both" ]]; then
    if [ ! -x "$STRASSEN_SEQ_EXE" ]; then
        echo "錯誤: 找不到可執行檔 $STRASSEN_SEQ_EXE" | tee -a $LOG_FILE
        exit 1
    fi
fi
if [[ "$MODE" == "par" || "$MODE" == "both" ]]; then
    if [ ! -x "$STRASSEN_PAR_EXE" ]; then
        echo "錯誤: 找不到可執行檔 $STRASSEN_PAR_EXE" | tee -a $LOG_FILE
        exit 1
    fi
fi

# ==========================================================
# 執行測試
# ==========================================================
for N in "${TEST_DIMENSIONS[@]}"; do
    echo -e "\n--- 測試 N=$N ---" | tee -a $LOG_FILE

    A_FILE="${DATA_DIR}/A_${N}"
    B_FILE="${DATA_DIR}/B_${N}"
    C_TRUE_FILE="${DATA_DIR}/C_${N}"

    if [ ! -f "$A_FILE" ] || [ ! -f "$B_FILE" ] || [ ! -f "$C_TRUE_FILE" ]; then
        echo "警告: 缺少資料檔 (A_${N}, B_${N}, C_${N})，跳過 N=$N" | tee -a $LOG_FILE
        continue
    fi

    # run sequential
    if [[ "$MODE" == "seq" || "$MODE" == "both" ]]; then
        echo "-> 執行 seq: $STRASSEN_SEQ_EXE $N $A_FILE $B_FILE $C_TRUE_FILE" | tee -a $LOG_FILE
        CMD_SEQ="${NVPROF:+$NVPROF }${STRASSEN_SEQ_EXE} \"$N\" \"$A_FILE\" \"$B_FILE\" \"$C_TRUE_FILE\""
        eval $CMD_SEQ 2>&1 | tee -a $LOG_FILE
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "警告: seq 在 N=$N 執行失敗 (退出碼 ${PIPESTATUS[0]})" | tee -a $LOG_FILE
        fi
    fi

    # run parallel (GPU) 
    if [[ "$MODE" == "par" || "$MODE" == "both" ]]; then
        echo "-> 執行 par: ${SRUN} ${NVPROF} ${STRASSEN_PAR_EXE} $N $A_FILE $B_FILE $C_TRUE_FILE" | tee -a $LOG_FILE
        # 如果 SRUN 為空字串則不使用 srun
        if [ -z "$SRUN" ]; then
            CMD_PAR="${NVPROF:+$NVPROF }${STRASSEN_PAR_EXE} \"$N\" \"$A_FILE\" \"$B_FILE\" \"$C_TRUE_FILE\""
        else
            CMD_PAR="${SRUN} ${NVPROF:+$NVPROF }${STRASSEN_PAR_EXE} \"$N\" \"$A_FILE\" \"$B_FILE\" \"$C_TRUE_FILE\""
        fi
        eval $CMD_PAR 2>&1 | tee -a $LOG_FILE
        # capture exit code of the executed program (before pipe)
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -ne 0 ]; then
            echo "警告: par 在 N=$N 執行失敗 (退出碼 $EXIT_CODE)" | tee -a $LOG_FILE
        fi
    fi

done

echo -e "\n--- 測試完成 ---" | tee -a $LOG_FILE
echo "結果已寫入 $LOG_FILE"

exit