@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PIPE=%ROOT%research_pipeline"
set "ART=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\artifacts\event_character_pair_specific"
set "LOGDIR=%ART%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%a-%%b-%%c"
for /f "tokens=1-3 delims=:., " %%a in ("%time%") do set "T=%%a%%b%%c"
set "LOG=%LOGDIR%\pair_specific_event_character_%D%_%T%.log"

echo ============================================================
echo Macro News FX - Pair-Specific Event Character Aware
echo Pipeline dir: %PIPE%
echo Log file: %LOG%
echo Steps: probability audit, pair-specific training, validation gate, OOS, report
echo Models: Elastic Net, Random Forest, CatBoost if installed
echo ============================================================
echo.

cd /d "%PIPE%"
if errorlevel 1 (
  echo Failed to enter pipeline dir: %PIPE%
  pause
  exit /b 1
)

call :run_step "66_audit_event_character_probabilities.py" "Audit probability scale"
if errorlevel 1 goto :failed
call :run_step "67_train_pair_specific_event_character_models.py" "Train pair-specific event-character models"
if errorlevel 1 goto :failed
call :run_step "68_pair_specific_validation_gate.py" "Select validation-gated pair models"
if errorlevel 1 goto :failed
call :run_step "69_test_pair_specific_event_character_oos.py" "Test selected pair models on OOS"
if errorlevel 1 goto :failed
call :run_step "70_make_pair_specific_event_character_report.py" "Write pair-specific report"
if errorlevel 1 goto :failed

echo.
echo Pair-specific event-character-aware pipeline finished.
echo Report: %ART%\reports\LAPORAN_PAIR_SPECIFIC_EVENT_CHARACTER_AWARE.md
echo Log: %LOG%
pause
exit /b 0

:run_step
set "SCRIPT=%~1"
set "TITLE=%~2"
echo ------------------------------------------------------------
echo [%date% %time%] START %TITLE%
echo Script: %SCRIPT%
echo ------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; & 'C:\Program Files\Python310\python.exe' -u '%SCRIPT%' 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
if errorlevel 1 (
  echo [%date% %time%] FAILED %TITLE%
  echo See log: %LOG%
  exit /b 1
)
echo [%date% %time%] DONE %TITLE%
exit /b 0

:failed
echo.
echo Pipeline failed.
echo See log: %LOG%
pause
exit /b 1
