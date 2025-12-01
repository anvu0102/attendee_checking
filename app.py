import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import zipfile
from deepface import DeepFace
import tempfile
import time 
import pandas as pd
# --- THƯ VIỆN GOOGLE DRIVE API VÀ OAUTH ---
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
# ----------------------------------------

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace (GDrive)",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Sử dụng ID Drive và OAuth Credentials từ st.secrets.")

# --- 2. Cấu hình & Hằng số (TẢI TỪ ST.SECRETS) ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# PHẠM VI (SCOPES) CHO GOOGLE DRIVE API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

# TẢI CÁC THÔNG TIN TỪ ST.SECRETS
try:
    GDRIVE_CLIENT_ID = st.secrets["GDRIVE_CLIENT_ID"]
    GDRIVE_CLIENT_SECRET = st.secrets["GDRIVE_CLIENT_SECRET"]
    GDRIVE_DATASET_FOLDER_ID = st.secrets["GDRIVE_DATASET_ID"] 
    GDRIVE_CHECKLIST_ID = st.secrets["GDRIVE_CHECKLIST_ID"]
    GDRIVE_NEW_DATA_FOLDER_ID = st.secrets["GDRIVE_NEW_DATA_ID"]
    # Khóa mới thêm vào để xác thực non-interactive
    GDRIVE_REFRESH_TOKEN = st.secrets["GDRIVE_REFRESH_TOKEN"] 
except KeyError as e:
    st.error(f"❌ Lỗi: Không tìm thấy khóa {e} trong st.secrets.")
    st.info("Vui lòng đảm bảo bạn đã định nghĩa tất cả các khóa (CLIENT_ID, CLIENT_SECRET, DATASET_ID, CHECKLIST_ID, NEW_DATA_ID, REFRESH_TOKEN) trong file .streamlit/secrets.toml hoặc trong giao diện Secrets của Streamlit Cloud.")
    st.stop()

# Các hằng số khác
DATASET_FOLDER = "dataset" 
CHECKLIST_FILENAME = "checklist.xlsx" 
CHECKLIST_SESSION_KEY = "attendance_df" 
DETECTOR_BACKEND = "opencv"

# ----------------------------------------------------------------------
#                             CÁC HÀM THỰC TẾ
# ----------------------------------------------------------------------

# --- 1. HÀM XÁC THỰC OAUTH (REAL - Non-interactive cho Streamlit Cloud) ---
@st.cache_resource(show_spinner="Đang tải và làm mới Access Token từ st.secrets...")
def get_valid_access_token_real(client_id, client_secret):
    """ 
    THỰC TẾ: Lấy Credentials từ st.secrets (dạng non-interactive) và làm mới token.
    """
    
    # 1. Tạo đối tượng Credentials từ Refresh Token
    creds = Credentials(
        token=None,
        refresh_token=GDRIVE_REFRESH_TOKEN,
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
            st.info("Gợi ý: Refresh Token có thể đã hết hạn hoặc bị thu hồi. Vui lòng kiểm tra lại token và quyền truy cập.")
            return None
    elif not creds.refresh_token:
        st.error("❌ Lỗi: Credentials không có Refresh Token.")
        return None
        
    st.success("✅ Access Token đã sẵn sàng.")
    return creds


# --- 2. HÀM TẢI FILE ĐƠN LẺ TỪ G-DRIVE (REAL) ---
# CHÚ Ý: Đã đổi tên tham số thành _credentials
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
        st.info(f"Đã tải thành công file: {output_filename}")
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi tải file {output_filename} từ Drive: {e}")
        st.warning("Gợi ý: Kiểm tra ID file và quyền truy cập của tài khoản đã xác thực.")
        return False


# --- 3. HÀM TẢI DATASET FOLDER (REAL) ---
@st.cache_resource(show_spinner="Đang tải Dataset Folder từ Google Drive...")
# CHÚ Ý: Đã đổi tên tham số thành _credentials
def download_dataset_folder_real(folder_id, target_folder, _credentials):
    """ THỰC TẾ: Tải toàn bộ nội dung folder Drive vào thư mục local. """
    if os.path.isdir(target_folder) and len(os.listdir(target_folder)) > 0:
        st.success(f"Dataset folder đã sẵn sàng tại '{target_folder}'. Bỏ qua tải xuống.")
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

        st.success(f"✅ Đã tải thành công {len(items)} file ảnh dataset vào thư mục '{target_folder}'.")
        return True
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tải Dataset Folder từ Drive: {e}")
        return False


# --- 4. HÀM UPLOAD FILE MỚI (REAL) ---
# CHÚ Ý: Đã đổi tên tham số thành _credentials
def upload_to_gdrive_real(file_path, drive_folder_id, drive_filename, _credentials):
    """
    Tải file lên Google Drive bằng Google Drive API, cần Credential thật.
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
        media = MediaFileUpload(file_path, mimetype='image/jpeg', resumable=True)
        
        with st.spinner(f"Đang tải file '{drive_filename}' lên Drive..."):
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        st.success(f"✅ **Upload Thành Công:** File '{drive_filename}' đã được lưu với ID: `{file.get('id')}`.")
        st.info(f"Đã lưu vào Drive Folder ID: **{drive_folder_id}**.")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Lỗi khi Upload file mới lên Drive: {e}")
        return False


# --- 5. HÀM TẢI CHECKLIST (REAL) ---
@st.cache_data(show_spinner="Đang tải và xử lý Checklist (XLSX) từ Google Drive...")
# CHÚ Ý: Đã đổi tên tham số thành _credentials
def load_checklist(file_id, filename, _credentials):
    """ Tải checklist XLSX và đọc thành DataFrame. """
    
    if not os.path.exists(filename):
        # Truyền _credentials vào hàm download
        download_file_from_gdrive(file_id, filename, _credentials)
        
    if os.path.exists(filename):
        try:
            # ĐỌC FILE XLSX
            df = pd.read_excel(filename) 
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

# ----------------------------------------------------------------------
#                             CÁC HÀM XỬ LÝ
# ----------------------------------------------------------------------

@st.cache_resource
def load_face_cascade(url, filename):
    """ Tải Haar Cascade cho OpenCV. """
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            classifier = cv2.CascadeClassifier(filename)
            if not classifier.empty():
                return classifier
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)


# --- Hàm Phát hiện Khuôn mặt (Giữ nguyên) ---
def detect_and_draw_face(image_bytes, cascade):
    """ Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh. """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, len(faces), image_bgr


# --- Hàm DeepFace Recognition (Giữ nguyên) ---
def verify_face_against_dataset(target_image_path, dataset_folder):
    """ Sử dụng DeepFace để so sánh ảnh đầu vào với dataset. """
    try:
        df_list = DeepFace.find(
            img_path=target_image_path, 
            db_path=dataset_folder, 
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True, 
            detector_backend=DETECTOR_BACKEND 
        )
        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            best_match = df_list[0].iloc[0]
            identity_path = best_match['identity']
            stt_match = os.path.splitext(os.path.basename(identity_path))[0] 
            distance = best_match['ArcFace_cosine'] 
            return stt_match, distance
        return None, None
    except Exception as e:
        if "Face could not be detected" in str(e):
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


# --- Logic Ghi Dữ Liệu và Lưu Ảnh Mới ---
# CHÚ Ý: Đã đổi tên tham số thành _credentials
def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, _credentials):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới lên Drive.
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return

    df = st.session_state[CHECKLIST_SESSION_KEY]
    
    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        try:
            stt_col = df.columns[0] 
            stt_match_str = str(stt_match).split('_')[0] 
            
            row_index = df[df[stt_col].astype(str).str.contains(stt_match_str, regex=False)].index
            
            if not row_index.empty:
                df.loc[row_index[0], session_name] = 'X'
                st.session_state[CHECKLIST_SESSION_KEY] = df 
                
                st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")
                st.info(f"⚠️ **Cần thêm chức năng ghi ngược (Write-Back) DataFrame này lên file XLSX Drive ID: {GDRIVE_CHECKLIST_ID}**.")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match_str}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp)
    else: 
        if 'new_data_counter' not in st.session_state:
            st.session_state['new_data_counter'] = 0
            
        st.session_state['new_data_counter'] += 1
        new_counter = st.session_state['new_data_counter']
        
        session_num = session_name.replace("Buổi ", "")
        drive_filename = f"B{session_num}_{new_counter}.jpg" 
        
        # --- TẠO FILE TẠM ĐỂ UPLOAD ---
        temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_UPLOAD_PATH = temp_file_for_upload.name
        temp_file_for_upload.close()
        
        try:
            image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
            
            # 2. Gọi hàm Upload Drive (REAL) - Truyền _credentials
            upload_to_gdrive_real(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, _credentials)

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH
# ----------------------------------------------------------------------

# LẤY CREDENTIALS ĐẦU TIÊN
CREDENTIALS = get_valid_access_token_real(GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET)

if not CREDENTIALS:
    st.error("❌ Không thể tiếp tục do không lấy được Credential hợp lệ.")
    st.stop()


# 7.1 KHỞI TẠO VÀ TẢI DATASET & CHECKLIST
# Tải Folder Dataset (REAL) - Truyền CREDENTIALS vào tham số _credentials
dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, CREDENTIALS) 
# Tải Checklist (XLSX) - Truyền CREDENTIALS vào tham số _credentials
checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, CREDENTIALS)

if checklist_df is not None:
    st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
    
st.markdown("---")

if not dataset_ready:
     st.warning("⚠️ Lỗi tải Dataset Folder. Vui lòng kiểm tra ID Drive Folder và quyền truy cập.")
     st.stop()
     
if checklist_df is None:
     st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
     st.stop()


st.info(f"Dataset đã sẵn sàng. Checklist có {len(checklist_df)} người.")


# 7.2 CHỌN BUỔI HỌC (Dropdown)
attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

if not attendance_cols:
     st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
     st.stop()

selected_session = st.selectbox(
    "1️⃣ **Chọn Buổi Điểm Danh**", 
    attendance_cols, 
    index=0,
    help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
)
st.success(f"Đang điểm danh cho: **{selected_session}**")

st.markdown("---")

# 7.3 CHỤP ẢNH VÀ XỬ LÝ
captured_file = st.camera_input("2️⃣ Chụp ảnh điểm danh:")

if captured_file is not None:
    
    image_bytes = captured_file.getvalue()
    
    with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
        
        processed_image_np, face_detected, num_faces, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
        processed_image = Image.fromarray(processed_image_np)
        
        # LƯU ẢNH TẠM THỜI cho DeepFace so khớp
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_IMAGE_PATH = temp_file.name
        temp_file.close() 
        
        cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
        
        # Thực hiện so khớp DeepFace
        stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

    # Xóa file tạm của DeepFace
    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)
        
    st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
    st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

    st.markdown("---")
    st.subheader("💡 Kết quả Điểm danh")
    
    if stt_match:
        st.balloons()
        st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
        st.markdown(f"""
        * **STT trùng khớp:** **{stt_match}**
        * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
        """)
        # Cập nhật checklist (truyền credentials)
        update_checklist_and_save_new_data(stt_match, selected_session, None, CREDENTIALS)
        
    elif face_detected and num_faces == 1:
        st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
        # Lưu ảnh mới (truyền image_bytes và credentials)
        update_checklist_and_save_new_data(None, selected_session, image_bytes, CREDENTIALS) 
        
    elif face_detected and num_faces > 1:
        st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

    else:
        st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
        st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

st.markdown("---")
st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
if CHECKLIST_SESSION_KEY in st.session_state:
    st.dataframe(st.session_state[CHECKLIST_SESSION_KEY])
