@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "BAT_DIR=%ROOT_DIR%BAT"
set "IND=%BAT_DIR%\_individual"

:menu
cls
echo ============================================================
echo Macro News FX - NewsEvents Pipeline Menu
echo ============================================================
echo Folder:
echo %ROOT_DIR%
echo.
echo MAIN TWO-STAGE PIPELINE
echo   1. Phase 2 - Audit label and timing
echo   2. Phase 2 - Build two-stage dataset
echo   3. Phase 3 - Train Stage 1 trade filter
echo   4. Phase 3 - Train Stage 2 direction model
echo   5. Phase 4 - Evaluate two-stage OOS
echo   6. Phase 5 - Pair-specific selection
echo   7. Phase 6 - Validation stability gate
echo   8. Phase 7 - Package selected two-stage policy
echo.
echo SVM / EBM EXTENSION
echo   9. Phase 8 - Exact SVM_RBF full to Phase 7
echo  10. Phase 8 - Exact SVM_RBF + EBM full to Phase 7
echo  11. Phase 8 - Train Exact SVM_RBF only
echo  12. Phase 8 - Train EBM only
echo  13. Phase 8 - Evaluate existing SVM/EBM to Phase 7
echo.
echo HYBRID MODEL EXPERIMENT
echo  23. CatBoost Block/Inverse hybrid
echo  24. Random Forest V2 base pair-specific
echo.
echo TEST / EXPORT
echo  14. Export MT5 signal table
echo  15. Test FX Backtrader
echo  16. Test ONNX parity
echo  17. Test Backtrader vs VectorBT
echo.
echo LEGACY / SUPPORT
echo  18. Manual block/inverse
echo  19. Train Event Study V3
echo  20. Test VectorBT Event Study V3
echo  21. Train model repair loop fast
echo  22. Train old EBM only
echo.
echo   0. Exit
echo ============================================================
echo.
set /p "CHOICE=Choose number: "

if "%CHOICE%"=="0" goto end
if "%CHOICE%"=="1" call "%IND%\RUN_PHASE2_LABEL_TIMING_AUDIT.bat" & goto menu
if "%CHOICE%"=="2" call "%IND%\RUN_PHASE2_BUILD_TWO_STAGE_DATASET.bat" & goto menu
if "%CHOICE%"=="3" call "%IND%\RUN_PHASE3_STAGE1_TRADE_FILTER.bat" & goto menu
if "%CHOICE%"=="4" call "%IND%\RUN_PHASE3_STAGE2_DIRECTION.bat" & goto menu
if "%CHOICE%"=="5" call "%IND%\RUN_PHASE4_TWO_STAGE_EVALUATION.bat" & goto menu
if "%CHOICE%"=="6" call "%IND%\RUN_PHASE5_PAIR_SPECIFIC_SELECTION.bat" & goto menu
if "%CHOICE%"=="7" call "%IND%\RUN_PHASE6_VALIDATION_STABILITY_GATE.bat" & goto menu
if "%CHOICE%"=="8" call "%IND%\RUN_PHASE7_PACKAGE_SELECTED_TWO_STAGE_POLICY.bat" & goto menu
if "%CHOICE%"=="9" call "%IND%\RUN_PHASE8_SVM_RBF_FULL_TO_PHASE7.bat" & goto menu
if "%CHOICE%"=="10" call "%IND%\RUN_PHASE8_SVM_EBM_FULL_TO_PHASE7.bat" & goto menu
if "%CHOICE%"=="11" call "%IND%\TRAIN_PHASE8_SVM_RBF_EXACT_TWO_STAGE.bat" & goto menu
if "%CHOICE%"=="12" call "%IND%\TRAIN_PHASE8_EBM_TWO_STAGE.bat" & goto menu
if "%CHOICE%"=="13" call "%IND%\RUN_PHASE8_SVM_EBM_TO_PHASE7.bat" & goto menu
if "%CHOICE%"=="14" call "%IND%\EXPORT_MT5_SIGNAL_TABLE.bat" & goto menu
if "%CHOICE%"=="15" call "%IND%\TEST_FX_BACKTRADER.bat" & goto menu
if "%CHOICE%"=="16" call "%IND%\TEST_FX_ONNX_PARITY.bat" & goto menu
if "%CHOICE%"=="17" call "%IND%\TEST_BACKTRADER_VS_VECTORBT.bat" & goto menu
if "%CHOICE%"=="18" call "%IND%\RUN_MANUAL_BLOCK_INVERSE.bat" & goto menu
if "%CHOICE%"=="19" call "%IND%\TRAIN_EVENT_STUDY_V3.bat" & goto menu
if "%CHOICE%"=="20" call "%IND%\TEST_VECTORBT_EVENT_STUDY_V3.bat" & goto menu
if "%CHOICE%"=="21" call "%IND%\TRAIN_MODEL_REPAIR_LOOP_FAST.bat" & goto menu
if "%CHOICE%"=="22" call "%IND%\TRAIN_EBM_ONLY.bat" & goto menu
if "%CHOICE%"=="23" call "%IND%\RUN_CATBOOST_BLOCK_INVERSE.bat" & goto menu
if "%CHOICE%"=="24" call "%IND%\RUN_RANDOM_FOREST_V2_BASE_PAIR_SPECIFIC.bat" & goto menu

echo.
echo Invalid choice: %CHOICE%
pause
goto menu

:end
echo Done.
