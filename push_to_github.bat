@echo off
set GIT="C:\Program Files\Git\bin\git.exe"

echo === Initializing Git repo ===
%GIT% init

echo === Configuring user ===
%GIT% config user.email "chandrakanth@proanalyst.com"
%GIT% config user.name "Chandrakanth"

echo === Setting default branch to main ===
%GIT% checkout -b main 2>nul || %GIT% checkout main 2>nul

echo === Staging all files ===
%GIT% add .

echo === Committing ===
%GIT% commit -m "Initial commit - ProAnalyst: RAG-based Job Description Analyzer"

echo === Setting remote ===
%GIT% remote remove origin 2>nul
%GIT% remote add origin https://github.com/MahadevJagtap/proanalyst.git

echo === Pushing to GitHub ===
%GIT% push -u origin main --force

echo === Done! ===
pause
