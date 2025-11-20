@echo off
REM Compilation script for paper_refactored.tex
REM Usage: compile.bat

echo =========================================
echo Compiling: paper_refactored.tex
echo =========================================

REM First pass
echo [1/4] First pdflatex pass...
pdflatex -interaction=nonstopmode paper_refactored.tex > nul 2>&1

REM BibTeX
echo [2/4] Running bibtex...
bibtex paper_refactored > nul 2>&1

REM Second pass
echo [3/4] Second pdflatex pass...
pdflatex -interaction=nonstopmode paper_refactored.tex > nul 2>&1

REM Third pass
echo [4/4] Third pdflatex pass...
pdflatex -interaction=nonstopmode paper_refactored.tex > nul 2>&1

REM Check if PDF was created
if exist paper_refactored.pdf (
    echo.
    echo ✓ Compilation successful!
    echo Output: paper_refactored.pdf
    echo.

    echo Quick check:
    echo - EU elasticity should be: -0.054 (p=0.171^)
    echo - EaP elasticity should be: -0.608*** (p^<0.001^)
    echo - Elasticity ratio should be: 11.3×
    echo - COVID ratio should be: 3.7×
    echo.
) else (
    echo.
    echo ✗ Compilation failed!
    echo Check the .log file for errors
    echo.
    exit /b 1
)

echo Done!
pause
