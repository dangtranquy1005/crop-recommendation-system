@echo off
chcp 65001 > nul
echo ===================================================
echo   Khoi chay Crop Recommendation System Web App
echo ===================================================

:: Them thu muc Python vao PATH neu chua co
set "PATH=C:\Users\Admin\AppData\Local\Programs\Python\Python313;C:\Users\Admin\AppData\Local\Programs\Python\Python313\Scripts;%PATH%"

python -m streamlit run app.py
pause

