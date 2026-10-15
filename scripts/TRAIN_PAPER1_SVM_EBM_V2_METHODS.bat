@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "PIPELINE_DIR=%ROOT_DIR%research_pipeline"
set "ARTIFACT_ROOT=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\artifacts"
set "LOG_DIR=%ARTIFACT_ROOT%\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "LOG_FILE=%LOG_DIR%\paper1_svm_ebm_v2_methods_%STAMP%.log"

if "%PAPER1_SVM_EBM_MODELS%"=="" set "PAPER1_SVM_EBM_MODELS=svm_rbf,ebm"
if "%PAPER1_SVM_MODE%"=="" set "PAPER1_SVM_MODE=approx"
if "%PAPER1_HEARTBEAT_SECONDS%"=="" set "PAPER1_HEARTBEAT_SECONDS=60"
set "PAPER1_VARIANTS=sparse_event_selection,jump_volatility_labeling,event_congestion_filter"

echo ============================================================
echo Paper 1 Macro News FX - SVM/EBM V2 Methods Training
echo Pipeline dir: %PIPELINE_DIR%
echo Log file: %LOG_FILE%
echo Models: %PAPER1_SVM_EBM_MODELS%
echo SVM mode: %PAPER1_SVM_MODE%
echo Variants: %PAPER1_VARIANTS%
echo.
echo V2 methods:
echo   - Sparse Event Selection
echo   - Jump / Volatility Labeling
echo   - Event Congestion Filter
echo.
echo Output:
echo   Common\Files\Dataset\artifacts\paper1_svm_ebm_base_and_v2_methods
echo ============================================================
echo.

if not exist "%PIPELINE_DIR%\94_paper1_train_svm_ebm_base_and_v2_methods.py" (
  echo ERROR: pipeline script not found.
  echo Missing: %PIPELINE_DIR%\94_paper1_train_svm_ebm_base_and_v2_methods.py
  pause
  exit /b 1
)

cd /d "%PIPELINE_DIR%"
if errorlevel 1 (
  echo ERROR: cannot enter pipeline directory.
  pause
  exit /b 1
)

echo [%date% %time%] START SVM/EBM V2 methods training
echo [batch] command: python -u 94_paper1_train_svm_ebm_base_and_v2_methods.py
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$env:PAPER1_SVM_EBM_MODELS='%PAPER1_SVM_EBM_MODELS%';" ^
  "$env:PAPER1_SVM_MODE='%PAPER1_SVM_MODE%';" ^
  "$env:PAPER1_HEARTBEAT_SECONDS='%PAPER1_HEARTBEAT_SECONDS%';" ^
  "$env:PAPER1_VARIANTS='%PAPER1_VARIANTS%';" ^
  "& python -u '94_paper1_train_svm_ebm_base_and_v2_methods.py' 2>&1 | Tee-Object -FilePath '%LOG_FILE%';" ^
  "exit $LASTEXITCODE"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo [%date% %time%] FAILED SVM/EBM V2 methods training
  echo See log: %LOG_FILE%
  pause
  exit /b %EXIT_CODE%
)

echo [%date% %time%] DONE SVM/EBM V2 methods training
echo See log: %LOG_FILE%
pause
exit /b 0
