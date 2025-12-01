# app.py
"""
File chính khởi chạy ứng dụng Streamlit.
"""
import streamlit as st
import sys
import os

# Import các hàm và biến từ config và check
from config import (
    GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN,
    get_valid_access_token_real,
    GDRIVE_DATASET_FOLDER_ID, GDRIVE_CHECKLIST_ID
)
# Sử dụng sys.path để đảm bảo có thể import check.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from check import main_app 

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh")
st.caption("Sử dụng ID Drive và OAuth Credentials từ st.secrets.")

# ----------------------------------------------------------------------
#                             LOGIC KHỞI CHẠY
# ----------------------------------------------------------------------

# LẤY CREDENTIALS ĐẦU TIÊN
# Hàm này được định nghĩa trong config.py và sử dụng st.cache_resource
CREDENTIALS = get_valid_access_token_real(
    GDRIVE_CLIENT_ID, 
    GDRIVE_CLIENT_SECRET, 
    GDRIVE_REFRESH_TOKEN
)

if not CREDENTIALS:
    st.error("❌ Không thể tiếp tục do không lấy được Credential hợp lệ. Vui lòng kiểm tra st.secrets.")
else:
    # Chạy giao diện chính
    main_app(CREDENTIALS)
