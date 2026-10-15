@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PIPE=%ROOT%research_pipeline"
set "ART=C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\artifacts\event_character_pair_specific\engine_validation"
set "LOGDIR=%ART%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%a-%%b-%%c"
for /f "tokens=1-3 delims=:., " %%a in ("%time%") do set "T=%%a%%b%%c"
set "LOG=%LOGDIR%\engine_validation_pair_specific_%D%_%T%.log"

echo ============================================================
echo Macro News FX - Engine Validation Pair-Specific Event Character
echo Pipeline dir: %PIPE%
echo Log file: %LOG%
echo Steps: export signals, VectorBT, Backtrader-style sequential, comparison report
echo ============================================================
echo.

cd /d "%PIPE%"
if errorlevel 1 (
  echo Failed to enter pipeline dir: %PIPE%
  pause
  exit /b 1
)

call :run_step "72_export_pair_specific_engine_signals.py" "Export final pair-specific signals"
if errorlevel 1 goto :failed
call :run_step "73_vectorbt_pair_specific_event_character_oos.py" "Run VectorBT validation"
if errorlevel 1 goto :failed
call :run_step "74_backtrader_pair_specific_event_character_oos.py" "Run Backtrader-style sequential validation"
if errorlevel 1 goto :failed
call :run_step "75_compare_pair_specific_engines.py" "Write engine comparison report"
if errorlevel 1 goto :failed

echo.
echo Engine validation finished.
echo Report: %ART%\LAPORAN_ENGINE_VALIDATION_PAIR_SPECIFIC_EVENT_CHARACTER.md
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
