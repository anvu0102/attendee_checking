# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
Đã bổ sung: Tích hợp streamlit-webrtc cho tính năng Auto Check (Live Stream).
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io 
import os
import tempfile
import pandas as pd
from deepface import DeepFace
import requests
import re 
import time
import datetime 

# THƯ VIỆN BỔ SUNG CHO GOOGLE DRIVE API
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- THƯ VIỆN BỔ SUNG CHO LIVESTREAM VÀ XỬ LÝ ĐA LUỒNG ---
from streamlit_webrtc import webrtc_stream, VideoTransformerBase
from typing import List
import threading 

# --- BIẾN TOÀN CỤC KIỂM SOÁT LUỒNG WEBRTC ---
lock = threading.Lock()
captured_frames: List[np.ndarray] = [] # Danh sách buffer ảnh đã chụp (BGR)
is_capturing = False # Cờ kiểm soát việc chụp ảnh/xử lý để tránh chụp trùng lặp
# -----------------------------------------------------------

# Import hằng số và hàm từ config.py (Đã sửa lỗi NameError)
from config import (
    HAAR_CASCADE_URL, CASCADE_FILENAME, 
    DATASET_FOLDER, CHECKLIST_FILENAME, CHECKLIST_SESSION_KEY, 
    DETECTOR_BACKEND, GDRIVE_CHECKLIST_ID, GDRIVE_NEW_DATA_FOLDER_ID,
    GDRIVE_DATASET_FOLDER_ID,
    download_file_from_gdrive, upload_to_gdrive_real, list_files_in_gdrive_folder,
    download_dataset_folder_real 
)


# ----------------------------------------------------------------------
#                             CÁC HÀM XỬ LÝ
# ----------------------------------------------------------------------

# --- LỚP XỬ LÝ KHUNG HÌNH (VIDEO TRANSFORMER) CHO WEBRTC ---
class FaceTrackingTransformer(VideoTransformerBase):
    """
    Xử lý từng khung hình: phát hiện khuôn mặt và chụp ảnh nếu Auto Check BẬT 
    VÀ chưa có ảnh nào đang được xử lý.
    """
    def __init__(self, face_cascade):
        self.face_cascade = face_cascade
        
    def transform(self, frame: np.ndarray) -> np.ndarray:
        global captured_frames, is_capturing
        
        # Chuyển đổi khung hình sang BGR (cho OpenCV)
        image = frame.to_ndarray(format="bgr24")
        
        # Phát hiện khuôn mặt
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        num_faces = len(faces)
        
        if num_faces == 1:
            with lock:
                # Chỉ chụp nếu chưa có ảnh nào đang chờ xử lý
                if not is_capturing:
                    
                    # TẠO KHUNG ĐỎ (Đã phát hiện và Chụp)
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4) # Màu đỏ
                    
                    # LƯU KHUNG HÌNH (BGR) để xử lý bên ngoài luồng webrtc
                    captured_frames.append(image) 
                    is_capturing = True
                    
                    # Hiển thị thông báo
                    cv2.putText(image, "CAPTURED! Processing...", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Đang xử lý, vẽ khung xanh (đã khóa)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Màu xanh
                        
        elif num_faces > 1:
            # Phát hiện nhiều khuôn mặt
            cv2.putText(image, f"Too many faces ({num_faces})!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Màu vàng
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Trả về khung hình đã xử lý
        return image 
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải Haar Cascade...")
def load_face_cascade(url, filename):
    """ Tải Haar Cascade cho OpenCV. (GIỮ NGUYÊN)"""
    try:
        if not os.path.exists(filename):
            r = requests.get(url)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(r.content)
            else:
                st.error(f"Lỗi tải file Haar Cascade: HTTP status {r.status_code}")
                return None

        classifier = cv2.CascadeClassifier(filename)
        if not classifier.empty():
            return classifier
        else:
            st.error("Lỗi: Khởi tạo Haar Cascade thất bại.")
            return None
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

# Load cascade ngay khi file được import
face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)


def detect_and_draw_face(image_bytes, cascade):
    """ 
    Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh. 
    (GIỮ NGUYÊN)
    """
    
    # Đọc ảnh từ bytes
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    # Lấy ảnh gốc BGR 
    image_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    
    # Tạo bản sao để vẽ khung
    image_bgr_with_frame = image_original_bgr.copy()
    
    gray = cv2.cvtColor(image_original_bgr, cv2.COLOR_RGB2GRAY)
    
    faces = []
    if cascade is not None:
        # Phát hiện khuôn mặt
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung lên bản sao
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB)

    # TRẢ VỀ: (ảnh có khung (RGB), ảnh GỐC (BGR), cờ phát hiện, số lượng khuôn mặt, TỌA ĐỘ KHUÔN MẶT)
    return processed_image_rgb, image_original_bgr, len(faces) > 0, len(faces), faces


def verify_face_against_dataset(target_image_path, dataset_folder):
    """ 
    Sử dụng DeepFace để so sánh ảnh đầu vào (ĐÃ CẮT) với dataset. 
    (GIỮ NGUYÊN)
    """
    try:
        # DeepFace.find trả về danh sách DataFrame, thường chỉ có 1
        df_list = DeepFace.find(
            img_path=target_image_path, 
            db_path=dataset_folder, 
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True, 
            detector_backend=DETECTOR_BACKEND 
        )
        
        # Kiểm tra nếu có kết quả và DataFrame đầu tiên không rỗng
        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            best_match = df_list[0].iloc[0]
            identity_path = best_match['identity']
            stt_match = os.path.splitext(os.path.basename(identity_path))[0].split('_')[0]
            distance = best_match['ArcFace_cosine'] 
            
            if pd.notna(distance):
                return stt_match, float(distance)
            else:
                st.error("❌ DeepFace không trả về độ tương đồng (distance) hợp lệ.")
                return None, None
                
        return None, None
    except Exception as e:
        if "Face could not be detected" in str(e):
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. (Kiểm tra chất lượng ảnh)")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


# BỎ DECORATOR @st.cache_data để buộc tải lại checklist mỗi khi app load
def load_checklist(file_id, filename, _credentials):
    """ 
    Tải checklist XLSX và đọc thành DataFrame. 
    (GIỮ NGUYÊN)
    """
    
    # 1. Tải file checklist mới nhất từ Drive (ghi đè lên file local nếu có)
    download_file_from_gdrive(file_id, filename, _credentials)
        
    # 2. Đọc file local vừa tải
    if os.path.exists(filename):
        try:
            # ĐỌC FILE XLSX
            df = pd.read_excel(filename) 
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

# --- CÁC HÀM XỬ LÝ DRIVE (GIỮ NGUYÊN) ---
def get_next_new_data_stt(_credentials):
    """
    Tìm số thứ tự lớn nhất trong folder NEW_DATA_FOLDER_ID trên Drive.
    """
    
    # 1. Lấy danh sách tên file từ Drive
    file_list = list_files_in_gdrive_folder(GDRIVE_NEW_DATA_FOLDER_ID, _credentials)
    
    max_stt = 0
    pattern = re.compile(r'B\d+_(\d+)\.jpe?g$', re.IGNORECASE)
    
    for filename in file_list:
        match = pattern.search(filename)
        if match:
            try:
                stt = int(match.group(1))
                if stt > max_stt:
                    max_stt = stt
            except ValueError:
                continue
    return max_stt + 1

def check_drive_file_existence(folder_id, filename, _credentials):
    """
    Kiểm tra xem file có tên filename đã tồn tại trong folder_id trên Drive hay chưa.
    """
    try:
        service = build('drive', 'v3', credentials=_credentials)
        query = (
            f"name='{filename}' and "
            f"'{folder_id}' in parents and "
            f"trashed=false"
        )
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        return len(items) > 0
    except Exception as e:
        st.error(f"❌ Lỗi Drive API khi kiểm tra file tồn tại: {e}")
        return False

@st.cache_resource(show_spinner="Đang kiểm tra/tạo folder Drive...")
def get_or_create_drive_folder(parent_id, folder_name, _credentials):
    """
    Tìm ID của folder con trong parent_id. Nếu chưa tồn tại, tạo mới.
    """
    try:
        service = build('drive', 'v3', credentials=_credentials)
        query = (
            f"mimeType='application/vnd.google-apps.folder' and "
            f"name='{folder_name}' and "
            f"'{parent_id}' in parents and "
            f"trashed=false"
        )
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            st.info(f"📁 Folder Drive: Đã tìm thấy '{folder_name}'.")
            return items[0]['id']
        else:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            file = service.files().create(body=file_metadata, fields='id').execute()
            st.success(f"📁 Folder Drive: Đã tạo folder mới '{folder_name}'.")
            return file.get('id')

    except Exception as e:
        st.error(f"❌ Lỗi Drive API khi kiểm tra/tạo folder: {e}")
        return None
        
def overwrite_gdrive_checklist_file(local_path, file_id, _credentials):
    pass

def load_dataset_image(stt_match, dataset_folder):
    """
    Tìm và trả về đường dẫn của ảnh dataset tương ứng với STT match đầu tiên.
    """
    pattern_simple = re.compile(rf'^{stt_match}\.jpe?g$', re.IGNORECASE)
    pattern_complex = re.compile(rf'^{stt_match}_.*\.jpe?g$', re.IGNORECASE)
    
    if os.path.isdir(dataset_folder):
        for filename in os.listdir(dataset_folder):
            if pattern_simple.match(filename):
                return os.path.join(dataset_folder, filename)
            if pattern_complex.match(filename):
                return os.path.join(dataset_folder, filename)
    return None
# --------------------------------------------------------------------------


# --- LOGIC GHI DỮ LIỆU VÀ LƯU ẢNH MỚI (GIỮ NGUYÊN) ---
def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, _credentials):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới lên Drive.
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return False 

    df = st.session_state[CHECKLIST_SESSION_KEY]
    updated = False 
    
    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        try:
            stt_col = df.columns[0] 
            row_index = df[df[stt_col].astype(str).str.contains(stt_match, regex=False)].index
            
            if not row_index.empty:
                
                # --- LƯU ẢNH GỐC VÀO FOLDER THEO BUỔI (Điểm danh thành công) ---
                stt = df.loc[row_index[0], stt_col]
                session_folder_name = session_name.replace("Buổi ", "B")
                target_folder_id = get_or_create_drive_folder(
                    GDRIVE_NEW_DATA_FOLDER_ID, 
                    session_folder_name, 
                    _credentials
                )
                
                if target_folder_id:
                    base_filename = f"{session_folder_name}_{stt}.jpg" 
                    drive_filename = base_filename 

                    if check_drive_file_existence(target_folder_id, base_filename, _credentials):
                        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                        drive_filename = f"{session_folder_name}_{stt}{timestamp}.jpg"
                        st.info(f"⚠️ File '{base_filename}' đã tồn tại. Đang lưu với tên mới: '{drive_filename}'.")
                    
                    temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    TEMP_UPLOAD_PATH = temp_file_for_upload.name
                    temp_file_for_upload.close()
                    
                    try:
                        image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
                        
                        upload_to_gdrive_real(TEMP_UPLOAD_PATH, target_folder_id, drive_filename, _credentials)
                        st.info(f"🖼️ Đã lưu ảnh thành công: {session_folder_name}/{drive_filename}")
                    
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lưu ảnh điểm danh thành công: {e}")
                    finally:
                        if os.path.exists(TEMP_UPLOAD_PATH):
                            os.remove(TEMP_UPLOAD_PATH)
                else:
                    st.warning("⚠️ Không thể xác định/tạo folder Drive để lưu ảnh.")
                # --------------------------------------------------------------------------

                if df.loc[row_index[0], session_name] != 'X':
                    df.loc[row_index[0], session_name] = 'X'
                    st.session_state[CHECKLIST_SESSION_KEY] = df 
                    updated = True 
                    st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")

                else:
                    st.info(f"Người có STT **{df.loc[row_index[0], stt_col]}** đã được điểm danh trong **{session_name}**.")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp) - SỬ DỤNG ẢNH GỐC
    else: 
        st.warning("⚠️ Đang lưu ảnh vào folder dữ liệu mới...")
        
        # --- LOGIC LƯU ẢNH GỐC KHÔNG KHỚP (GIỮ NGUYÊN) ---
        next_counter = get_next_new_data_stt(_credentials)
        session_num = session_name.replace("Buổi ", "")
        drive_filename = f"B{session_num}_{next_counter}.jpg" 
        
        temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_UPLOAD_PATH = temp_file_for_upload.name
        temp_file_for_upload.close()
        
        try:
            image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
            
            upload_to_gdrive_real(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, _credentials)
            st.info(f"🖼️ Đã lưu ảnh không khớp vào folder chung: {drive_filename}")

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)
        # ----------------------------------------------------------
                
    return updated 


# --- HÀM: CẬP NHẬT PLACEHOLDER CHECKLIST (GIỮ NGUYÊN) ---
def update_checklist_display(checklist_placeholder, current_df):
    """Cập nhật nội dung của placeholder checklist."""
    with checklist_placeholder.container():
        st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
        st.dataframe(current_df)
        
        output = io.BytesIO()
        current_df.to_excel(output, index=False, sheet_name='Checklist_Cap_Nhat')
        excel_data = output.getvalue()
        
        st.download_button(
            label="⬇️ Tải file Excel Checklist đã cập nhật",
            data=excel_data,
            file_name="Checklist_DiemDanh_CapNhat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Tải về file Excel (XLSX) chứa dữ liệu điểm danh mới nhất trong phiên làm việc hiện tại."
        )
# -----------------------------------------------


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    global captured_frames, is_capturing
    
    # === KHỞI TẠO KEY SESSION STATE ===
    if 'camera_input_key' not in st.session_state:
        st.session_state['camera_input_key'] = 0
        
    if 'auto_check_enabled' not in st.session_state:
        st.session_state['auto_check_enabled'] = False
    # =================================

    # 1. Tải Dataset & Checklist
    dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials) 
    
    if CHECKLIST_SESSION_KEY not in st.session_state:
        checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)
        if checklist_df is not None:
            st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
        else:
            st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
            return

    checklist_df = st.session_state[CHECKLIST_SESSION_KEY]
        
    st.markdown("---")

    checklist_placeholder = st.empty()
    
    st.markdown("---") 

    if not dataset_ready:
         st.warning("⚠️ Lỗi tải Dataset Folder. Vui lòng kiểm tra ID Drive Folder và quyền truy cập.")
         return
         
    if checklist_df is None:
         st.warning("⚠️ Checklist hiện tại không hợp lệ (Kiểm tra lỗi tải lần đầu).")
         return

    st.info(f"Checklist có {len(checklist_df)} người.")

    # 2. Chọn Buổi Học (Dropdown)
    attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

    if not attendance_cols:
         st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
         return

    display_options = ["--- Vui lòng chọn buổi ---"] + attendance_cols
    
    selected_session_display = st.selectbox(
        "Chọn Buổi điểm danh", 
        display_options, 
        index=0, 
        help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
    )
    
    selected_session = selected_session_display if selected_session_display != "--- Vui lòng chọn buổi ---" else None

    # --- BỔ SUNG: CHECKBOX HIỂN THỊ ẢNH DEBUG VÀ AUTO CHECK ---
    col_debug, col_auto = st.columns([0.7, 0.3])
    
    with col_debug:
        show_debug_images = st.checkbox(
            "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
            value=True, 
            help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset."
        )
        
    with col_auto:
        auto_check = st.checkbox(
            "Auto Check (Live)",
            value=st.session_state['auto_check_enabled'],
            key='auto_check_checkbox', 
            help="Bật để kích hoạt livestream điểm danh tự động (yêu cầu thư viện streamlit-webrtc)."
        )
        st.session_state['auto_check_enabled'] = auto_check

    st.markdown("---")

    # 3. Xử Lý Chụp Ảnh
    result_placeholder = st.empty()
    captured_file_bgr = None # Biến để lưu ảnh chụp từ webrtc hoặc camera_input
    
    # --- LOGIC ĐIỂM DANH LIVE VỚI STREAMLIT-WEBRTC ---
    if selected_session and auto_check:
        st.subheader("🔴 Đang Live: Auto Check (Phát hiện 1 khuôn mặt để chụp)")
        
        # Khởi tạo Stream
        webrtc_ctx = webrtc_stream(
            key="face-tracking-stream",
            video_processor_factory=lambda: FaceTrackingTransformer(face_cascade),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )
        
        # KIỂM TRA NẾU CÓ KHUNG HÌNH ĐƯỢC CHỤP TỪ WEBRTC
        if captured_frames:
            with lock:
                # Lấy khung hình đầu tiên và xóa khỏi danh sách
                captured_file_bgr = captured_frames.pop(0) 
            
            # Khung hình đã được chụp, tiếp tục xuống khối xử lý ảnh chung
            st.warning("🔔 Ảnh đã chụp! Đang xử lý DeepFace...")
            # Không cần rerun ở đây, vì việc xử lý ảnh sẽ tự nhiên xảy ra
            # và nếu auto check, nó sẽ rerun ở cuối khối xử lý
            
        elif webrtc_ctx.state.playing:
             st.info("⚠️ Vui lòng nhìn thẳng vào camera. Đảm bảo chỉ có 1 khuôn mặt trong khung hình.")
        

    # --- LOGIC CHỤP ẢNH TĨNH VỚI st.camera_input ---
    elif selected_session and not auto_check: 
        st.subheader("📸 Chụp Ảnh Tĩnh (Thủ công)")
        
        captured_file = st.camera_input(
            "Chụp ảnh điểm danh", 
            key=f"camera_input_{st.session_state['camera_input_key']}" 
        )
        
        if captured_file is not None:
             # Đọc ảnh tĩnh thành bytes
            image_bytes_original = captured_file.getvalue() 
            # Chuyển đổi bytes sang BGR
            image_pil = Image.open(io.BytesIO(image_bytes_original)).convert('RGB')
            captured_file_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            # Khối này sẽ tiếp tục xuống xử lý ảnh chung

    
    # ----------------------------------------------------------------------
    # --- LOGIC XỬ LÝ ẢNH CHUNG (Áp dụng cho cả Live và Chụp tĩnh) ---
    # ----------------------------------------------------------------------
    if captured_file_bgr is not None:
        
        stt_match = None
        distance = None
        TEMP_IMAGE_PATH = None
        
        # Chuyển ảnh BGR (từ webrtc hoặc camera_input) sang bytes RGB để xử lý DeepFace/Drive
        image_bytes_original_rgb = io.BytesIO()
        # Chuyển BGR -> RGB -> PIL Image -> Bytes
        Image.fromarray(cv2.cvtColor(captured_file_bgr, cv2.COLOR_BGR2RGB)).save(image_bytes_original_rgb, format='JPEG')
        image_bytes_original = image_bytes_original_rgb.getvalue()

        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            
            # Bắt buộc phải phát hiện lại khuôn mặt để lấy tọa độ chính xác cho việc cắt
            processed_image_np, image_original_bgr, face_detected, num_faces, faces = detect_and_draw_face(image_bytes_original, face_cascade)
            processed_image = Image.fromarray(processed_image_np)
            
            # Kiểm tra chỉ có 1 khuôn mặt và tiến hành cắt
            if face_detected and num_faces == 1:
                (x, y, w, h) = faces[0]
                
                # TĂNG KÍCH THƯỚC KHUNG (Padding 20%)
                padding = int(0.2 * w)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_original_bgr.shape[1], x + w + padding)
                y2 = min(image_original_bgr.shape[0], y + h + padding)

                cropped_face_bgr = image_original_bgr[y1:y2, x1:x2]
                
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                TEMP_IMAGE_PATH = temp_file.name
                temp_file.close() 
                cv2.imwrite(TEMP_IMAGE_PATH, cropped_face_bgr)
                
                # Thực hiện so khớp DeepFace trên ảnh đã cắt
                stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)
            
        
        # HIỂN THỊ KẾT QUẢ TRONG PLACEHOLDER
        with result_placeholder.container():
            st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
            st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

            st.markdown("---")
            st.subheader("💡 Kết quả Điểm danh")
            
            # -------------------------- TRƯỜNG HỢP 1: THÀNH CÔNG --------------------------
            if stt_match and distance is not None: 
                st.balloons()
                st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
                
                if show_debug_images: 
                    dataset_image_path = load_dataset_image(stt_match, DATASET_FOLDER)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if TEMP_IMAGE_PATH:
                            st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=True)
                    with col2:
                        if dataset_image_path:
                            st.image(dataset_image_path, caption=f"Dataset (STT: {stt_match})", use_column_width=True)
                        else:
                            st.warning("Không tìm thấy ảnh dataset để hiển thị.")
                
                st.markdown(f"""
                * **STT trùng khớp:** **{stt_match}**
                * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
                """)
                
                # Cập nhật checklist VÀ LƯU ẢNH GỐC THÀNH CÔNG
                update_checklist_and_save_new_data(stt_match, selected_session, image_bytes_original, credentials)
                
            # -------------------------- TRƯỜNG HỢP 2: KHÔNG KHỚP --------------------------
            elif face_detected and num_faces == 1:
                st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
                
                if show_debug_images: 
                    if TEMP_IMAGE_PATH:
                        st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=False)
                
                # Lưu ảnh gốc
                update_checklist_and_save_new_data(None, selected_session, image_bytes_original, credentials) 
                
            # -------------------------- TRƯỜNG HỢP 3: NHIỀU KHUÔN MẶT/KHÔNG PHÁT HIỆN --------------------------
            elif face_detected and num_faces > 1:
                st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

            else:
                st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
                st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

            # --- LOGIC TỰ ĐỘNG CLEAR VÀ RERUN (CHỈ KHI AUTO CHECK BẬT) ---
            if auto_check:
                
                # Cập nhật checklist display trước khi rerun 
                if CHECKLIST_SESSION_KEY in st.session_state:
                     update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
                     
                time.sleep(5) # Đợi 5 giây 
                
                with lock:
                    # Mở lại cờ cho phép chụp ảnh trong luồng video
                    is_capturing = False 
                
                # Xóa file tạm
                if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                    os.remove(TEMP_IMAGE_PATH)
                    
                st.rerun() # Buộc rerun để khởi động lại luồng webrtc

            # --- Dọn dẹp cho chế độ chụp tĩnh (Manual Check) ---
            elif not auto_check:
                 if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                    os.remove(TEMP_IMAGE_PATH)
                    
                 # Reset camera input key để nó hiện lại nút "Take Photo"
                 st.session_state['camera_input_key'] += 1
                 
                 # Cập nhật checklist trước khi reset
                 if CHECKLIST_SESSION_KEY in st.session_state:
                     update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
                 st.rerun() 
            # -------------------------------------------------------------
            
    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU
    if CHECKLIST_SESSION_KEY in st.session_state:
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
