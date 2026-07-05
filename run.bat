@echo off
REM ============================================
REM  Nowcasting Public — Windows стартиране
REM ============================================
REM  Стартирай от Anaconda Prompt!
REM ============================================

echo.
echo ========================================
echo  NOWCASTING PUBLIC — Bulgaria
echo ========================================
echo.

REM Активирай conda environment (ако не е активен)
call conda activate nowcast 2>nul

REM Стартирай pipeline
python "%~dp0run_nowcast.py" %*

echo.
echo Готово! Картите са в data\output\maps\
pause
