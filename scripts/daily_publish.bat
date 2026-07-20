@echo off
REM ============================================================================
REM  Daily publish: process every recording in gameplay/incoming/ and upload
REM  the merged long-form video plus one short per match to YouTube.
REM
REM  Run by Windows Task Scheduler once a day. Register it with:
REM
REM    schtasks /Create /TN "CR Daily Publish" /TR "\"<repo>\scripts\daily_publish.bat\"" ^
REM             /SC DAILY /ST 09:00 /RL LIMITED /F
REM
REM  WHY 09:00: the run takes ~50 minutes (about 31 processing + 15 uploading),
REM  so a morning start leaves the videos waiting privately in Studio well before
REM  the evening, whenever you get round to reviewing and publishing them.
REM
REM  Exit code 0 = success. Anything else is a failure worth looking at; the
REM  full log for the run is written under logs/.
REM
REM  PRIVACY: uploads land PRIVATE and stay private. Nothing is auto-published;
REM  you make videos public yourself in YouTube Studio. Add --schedule to hand
REM  that decision to the configured IST slots instead.
REM ============================================================================

setlocal

REM Resolve the repo root from this script's own location, so the task works
REM regardless of the working directory Task Scheduler happens to start in.
set "REPO=%~dp0.."
pushd "%REPO%" || exit /b 1

set "PROFILE=iphone_16_pro_max"
set "PRIVACY=private"

if not exist "logs" mkdir "logs"
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "STAMP=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%_%DT:~8,2%%DT:~10,2%"
set "LOG=logs\daily_publish_%STAMP%.log"

echo [%date% %time%] starting daily publish >> "%LOG%"

REM No recording argument: process the whole inbox, archiving each when done.
REM --cleanup-raw deletes ~6.8 GB per session (clips, merge, shorts, frame cache
REM AND the original recording) once every video has uploaded. It is skipped
REM automatically if any upload failed, so a partial run never destroys footage
REM that never got published.
".venv\Scripts\python.exe" -m backend.main auto ^
    --profile "%PROFILE%" ^
    --upload ^
    --privacy "%PRIVACY%" ^
    --cleanup-raw >> "%LOG%" 2>&1

set "RC=%ERRORLEVEL%"
echo [%date% %time%] finished with exit code %RC% >> "%LOG%"

REM Prune logs older than 30 days. The media is deleted each run, so logs are
REM the only thing that would otherwise grow without bound.
forfiles /P "logs" /M "daily_publish_*.log" /D -30 /C "cmd /c del @path" 2>nul

popd
exit /b %RC%
