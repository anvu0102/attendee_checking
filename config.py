# config.py
"""
Chứa các hằng số cấu hình, biến toàn cục, và các hàm
liên quan đến Google Drive API và xác thực.
"""
import streamlit as st
import os
import requests
import re # Thêm thư viện re để xử lý regex
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ----------------------------------------------------------------------
#                             HẰNG SỐ CẤU HÌNH
# ----------------------------------------------------------------------

# Cấu hình OpenCV
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# PHẠM VI (SCOPES) CHO GOOGLE DRIVE API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

# Các hằng số khác
DATASET_FOLDER = "dataset" 
CHECKLIST_FILENAME = "checklist.xlsx" 
CHECKLIST_SESSION_KEY = "attendance_df" 
DETECTOR_BACKEND = "yolo" #change to opencv, retinaface, mtcnn for comparison

# ----------------------------------------------------------------------
#                             TẢI THÔNG TIN TỪ ST.SECRETS
# ----------------------------------------------------------------------

try:
    GDRIVE_CLIENT_ID = st.secrets["GDRIVE_CLIENT_ID"]
    GDRIVE_CLIENT_SECRET = st.secrets["GDRIVE_CLIENT_SECRET"]
    GDRIVE_DATASET_FOLDER_ID = st.secrets["GDRIVE_DATASET_ID"] 
    GDRIVE_CHECKLIST_ID = st.secrets["GDRIVE_CHECKLIST_ID"]
    GDRIVE_NEW_DATA_FOLDER_ID = st.secrets["GDRIVE_NEW_DATA_ID"]
    GDRIVE_REFRESH_TOKEN = st.secrets["GDRIVE_REFRESH_TOKEN"] 
except KeyError as e:
    st.error(f"❌ Lỗi: Không tìm thấy khóa {e} trong st.secrets.")
    st.info("Vui lòng đảm bảo bạn đã định nghĩa tất cả các khóa (CLIENT_ID, CLIENT_SECRET, DATASET_ID, CHECKLIST_ID, NEW_DATA_ID, REFRESH_TOKEN) trong file .streamlit/secrets.toml.")
    # Sử dụng st.stop() thay vì sys.exit() để dừng ứng dụng Streamlit
    st.stop() 


# ----------------------------------------------------------------------
#                             CÁC HÀM XÁC THỰC & DRIVE
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải và làm mới Access Token từ st.secrets...")
def get_valid_access_token_real(client_id, client_secret, refresh_token):
    """ 
    Lấy Credentials từ st.secrets (dạng non-interactive) và làm mới token.
    """
    
    # 1. Tạo đối tượng Credentials từ Refresh Token
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES
    )

    # 2. Làm mới Token
    if creds.expired and creds.refresh_token:
        try:
            st.info("Đang làm mới Access Token...")
            creds.refresh(Request())
            st.success("✅ Đã làm mới Access Token thành công.")
        except Exception as e:
            st.error(f"❌ Lỗi khi làm mới Access Token: {e}")
            st.info("Gợi ý: Refresh Token có thể đã hết hạn hoặc bị thu hồi.")
            return None
    elif not creds.refresh_token:
        st.error("❌ Lỗi: Credentials không có Refresh Token.")
        return None
        
    # st.success("✅ Access Token is ready.")
    return creds


def download_file_from_gdrive(file_id, output_filename, _credentials):
    """ Tải file từ Google Drive dùng Google Drive API. """
    
    try:
        service = build('drive', 'v3', credentials=_credentials)
        request = service.files().get_media(fileId=file_id)
        
        with open(output_filename, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            with st.spinner(f"Đang tải file {output_filename}..."):
                while done is False:
                    status, done = downloader.next_chunk()
        st.info(f"File successfully downloaded: {output_filename}")
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi tải file {output_filename} từ Drive: {e}")
        st.warning("Gợi ý: Kiểm tra ID file và quyền truy cập của tài khoản đã xác thực.")
        return False


@st.cache_resource(show_spinner="Đang tải Dataset Folder từ Google Drive...")
def download_dataset_folder_real(folder_id, target_folder, _credentials):
    """ Tải toàn bộ nội dung folder Drive vào thư mục local. """
    if os.path.isdir(target_folder) and len(os.listdir(target_folder)) > 0:
        # st.success(f"✅ Dataset folder is ready at '{target_folder}'. Download skipped.")
        return True

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    try:
        service = build('drive', 'v3', credentials=_credentials)
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query, 
            pageSize=1000,
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])

        if not items:
            st.warning(f"Folder ID: {folder_id} trống rỗng. Không có dataset.")
            return False

        st.info(f"Tìm thấy {len(items)} file trong dataset. Đang tải xuống...")
        
        for item in items:
            file_id = item['id']
            file_name = item['name']
            output_path = os.path.join(target_folder, file_name)
            
            request = service.files().get_media(fileId=file_id)
            with open(output_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

        # st.success(f"✅ Đã tải thành công {len(items)} file ảnh dataset vào thư mục '{target_folder}'.")
        return True
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tải Dataset Folder từ Drive: {e}")
        return False

# --- HÀM MỚI: LIỆT KÊ FILE TRONG FOLDER DRIVE ---
@st.cache_data(ttl=3600) # Cache kết quả trong 1 giờ
def list_files_in_gdrive_folder(folder_id, _credentials):
    """
    Liệt kê tất cả tên file trong một folder Drive chỉ định.
    Trả về danh sách các tên file (string).
    """
    if _credentials is None:
        st.error("❌ Lỗi Auth: Không thể liệt kê file vì không có Credential hợp lệ.")
        return []
    
    try:
        service = build('drive', 'v3', credentials=_credentials)
        
        # Chỉ lấy tên file, giới hạn 1000 file (pageSize=1000)
        query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
        results = service.files().list(
            q=query, 
            pageSize=1000,
            fields="nextPageToken, files(name)"
        ).execute()
        
        items = results.get('files', [])
        
        # Trả về danh sách tên file
        return [item['name'] for item in items]
        
    except Exception as e:
        st.error(f"❌ Lỗi khi liệt kê file trong thư mục Drive ID {folder_id}: {e}")
        return []

def upload_to_gdrive_real(file_path, drive_folder_id, drive_filename, _credentials):
    """
    Tải file lên Google Drive bằng Google Drive API.
    """
    if _credentials is None:
        st.error("❌ Lỗi Auth: Không thể upload vì không có Credential hợp lệ.")
        return False
    
    try:
        service = build('drive', 'v3', credentials=_credentials)
        
        # Metadata của file
        file_metadata = {
            'name': drive_filename,
            'parents': [drive_folder_id] 
        }
        
        # Media to upload
        # Thay đổi mimetype tùy thuộc vào loại file bạn đang upload (ở đây mặc định là JPEG)
        media = MediaFileUpload(file_path, mimetype='image/jpeg', resumable=True)
        
        with st.spinner(f"Đang tải file '{drive_filename}' lên Drive..."):
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        # st.success(f"✅ **Upload Thành Công:** File '{drive_filename}' đã được lưu.")
        # st.info(f"Đã lưu vào Drive Folder ID: **{drive_folder_id}**.")
        
        # Xóa cache của hàm list_files_in_gdrive_folder để dữ liệu mới được cập nhật
        list_files_in_gdrive_folder.clear()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Lỗi khi Upload file mới lên Drive: {e}")
        return False
