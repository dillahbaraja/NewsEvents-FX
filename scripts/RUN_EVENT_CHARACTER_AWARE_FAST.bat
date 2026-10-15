@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PIPE=%ROOT%research_pipeline"
set "ART=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\artifacts\event_character_aware"
set "LOGDIR=%ART%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%a-%%b-%%c"
for /f "tokens=1-3 delims=:., " %%a in ("%time%") do set "T=%%a%%b%%c"
set "LOG=%LOGDIR%\event_character_aware_fast_%D%_%T%.log"

echo ============================================================
echo Macro News FX - Event Character Aware FAST Pipeline
echo Pipeline dir: %PIPE%
echo Log file: %LOG%
echo Steps: dataset, event profiles, clusters, EN/RF/CatBoost validation, report
echo Period: train 2010-2020, validation 2021-2023, OOS remains untouched
echo ============================================================
echo.

cd /d "%PIPE%"
if errorlevel 1 (
  echo Failed to enter pipeline dir: %PIPE%
  pause
  exit /b 1
)

call :run_step "60_build_event_character_dataset.py" "Build event-aware dataset"
if errorlevel 1 goto :failed
call :run_step "61_build_event_profiles.py" "Build event profiles"
if errorlevel 1 goto :failed
call :run_step "62_cluster_event_profiles.py" "Cluster event profiles"
if errorlevel 1 goto :failed
call :run_step "63_train_event_character_aware_fast_models.py" "Train fast event-character-aware models"
if errorlevel 1 goto :failed
call :run_step "65_test_event_character_aware_oos.py" "Test event-character-aware models on OOS datatest"
if errorlevel 1 goto :failed
call :run_step "64_make_event_character_aware_report.py" "Write event-character-aware report"
if errorlevel 1 goto :failed

echo.
echo Event Character Aware FAST pipeline finished.
echo Report: %ART%\reports\LAPORAN_EVENT_CHARACTER_AWARE.md
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
