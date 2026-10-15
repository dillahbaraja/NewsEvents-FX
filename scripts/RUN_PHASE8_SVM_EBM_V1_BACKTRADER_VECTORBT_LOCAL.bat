@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "PIPELINE_DIR=%ROOT_DIR%research_pipeline"
set "DATA_DIR=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset"
set "ARTIFACT_DIR=%DATA_DIR%\artifacts"
set "LOG_DIR=%ARTIFACT_DIR%\logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "LOG_FILE=%LOG_DIR%\phase8_svm_ebm_v1_backtrader_vectorbt_local_%STAMP%.log"

if "%PHASE8_EVAL_MODELS%"=="" set "PHASE8_EVAL_MODELS=svm_rbf_exact_two_stage,ebm_two_stage"
set "PHASE7_POLICY_DIR=%ARTIFACT_DIR%\phase8_svm_ebm_phase7_selected_policy"
set "PHASE7_BACKTRADER_OUT_DIR=%ARTIFACT_DIR%\backtrader_phase8_svm_ebm_v1_policy_comparison"
set "PHASE7_VECTORBT_OUT_DIR=%ARTIFACT_DIR%\vectorbt_phase8_svm_ebm_v1_policy_comparison"

echo ============================================================
echo Phase 8 SVM/EBM V1 - Local Backtrader + VectorBT Test
echo Pipeline dir: %PIPELINE_DIR%
echo Data dir: %DATA_DIR%
echo Models: %PHASE8_EVAL_MODELS%
echo Log file: %LOG_FILE%
echo.
echo Steps:
echo   1. Phase 8 OOS evaluation + Phase 7 policy package
echo   2. Backtrader sequential test
echo   3. VectorBT portfolio/event-audit test
echo.
echo Note:
echo   Exact SVM can be very slow on local Windows.
echo   EBM needs working interpret/libebm runtime.
echo ============================================================
echo.

if not exist "%PIPELINE_DIR%\47_phase8_run_svm_ebm_to_phase7.py" (
  echo ERROR: pipeline script not found.
  echo Missing: %PIPELINE_DIR%\47_phase8_run_svm_ebm_to_phase7.py
  pause
  exit /b 1
)

cd /d "%PIPELINE_DIR%"
if errorlevel 1 (
  echo ERROR: cannot enter pipeline directory.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Continue'; $code=0; & {" ^
  "  $env:PHASE8_EVAL_MODELS='%PHASE8_EVAL_MODELS%';" ^
  "  $env:PHASE7_POLICY_DIR='%PHASE7_POLICY_DIR%';" ^
  "  $env:PHASE7_BACKTRADER_OUT_DIR='%PHASE7_BACKTRADER_OUT_DIR%';" ^
  "  $env:PHASE7_VECTORBT_OUT_DIR='%PHASE7_VECTORBT_OUT_DIR%';" ^
  "  Write-Host '[STEP 1/3] Phase 8 model OOS evaluation + Phase 7 policy packaging';" ^
  "  python -u '47_phase8_run_svm_ebm_to_phase7.py';" ^
  "  if ($LASTEXITCODE -ne 0) { $script:code=$LASTEXITCODE; return }" ^
  "  Write-Host ''; Write-Host '[STEP 2/3] Backtrader sequential test';" ^
  "  python -u '48_backtrader_compare_phase7_selected_policy.py';" ^
  "  if ($LASTEXITCODE -ne 0) { $script:code=$LASTEXITCODE; return }" ^
  "  Write-Host ''; Write-Host '[STEP 3/3] VectorBT portfolio/event-audit test';" ^
  "  python -u '95_vectorbt_compare_phase7_selected_policy.py';" ^
  "  if ($LASTEXITCODE -ne 0) { $script:code=$LASTEXITCODE; return }" ^
  "} 2>&1 | Tee-Object -FilePath '%LOG_FILE%'; exit $code"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo FAILED Phase 8 SVM/EBM V1 local test.
  echo See log: %LOG_FILE%
  pause
  exit /b %EXIT_CODE%
)

echo DONE Phase 8 SVM/EBM V1 local test.
echo.
echo Outputs:
echo   %PHASE7_POLICY_DIR%
echo   %PHASE7_BACKTRADER_OUT_DIR%
echo   %PHASE7_VECTORBT_OUT_DIR%
echo.
echo Log:
echo   %LOG_FILE%
pause
exit /b 0
