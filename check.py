import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io 
import os
import tempfile
import pandas as pd
from deepface import DeepFace
import time
import datetime 

# THƯ VIỆN BỔ SUNG CHO GOOGLE DRIVE API
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Import hằng số và hàm từ config.py
from config import (
    DATASET_FOLDER, CHECKLIST_FILENAME, CHECKLIST_SESSION_KEY, 
    DETECTOR_BACKEND, GDRIVE_CHECKLIST_ID, GDRIVE_DATASET_FOLDER_ID,
    GDRIVE_NEW_DATA_FOLDER_ID,
    download_file_from_gdrive, upload_to_gdrive_real, list_files_in_gdrive_folder
)

# --- 1. CÁC HÀM XỬ LÝ NHẬN DIỆN ---

def robust_detect_and_crop_face(image_bytes):
    """Phát hiện và cắt khuôn mặt bằng DeepFace."""
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    image_bgr_with_frame = image_original_bgr.copy()
    
    face_detected, num_faces, temp_cropped_path = False, 0, None
    
    try:
        faces_extracted = DeepFace.extract_faces(
            img_path=image_np, 
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=False
        )
        num_faces = len(faces_extracted)
        face_detected = num_faces > 0

        if num_faces == 1:
            facial_area = faces_extracted[0]['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            padding = int(0.2 * w)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image_original_bgr.shape[1], x + w + padding), min(image_original_bgr.shape[0], y + h + padding)
            cropped_face_bgr = image_original_bgr[y1:y2, x1:x2]
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_cropped_path = temp_file.name
            temp_file.close() 
            cv2.imwrite(temp_cropped_path, cropped_face_bgr)
            
    except Exception: pass
    return cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB), face_detected, num_faces, temp_cropped_path

def verify_face_against_dataset(target_image_path, dataset_folder):
    """So khớp ảnh đã cắt với database local."""
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
            # Lấy tên file để trích xuất STT
            identity_path = best_match['identity']
            stt_match = os.path.splitext(os.path.basename(identity_path))[0].split('_')[0]
            return stt_match, float(best_match['ArcFace_cosine'])
        return None, None
    except Exception: return None, None

# --- 2. CÁC HÀM XỬ LÝ DỮ LIỆU DRIVE & CHECKLIST ---

def load_checklist(file_id, filename, _credentials):
    """Tải và đọc file Excel checklist."""
    download_file_from_gdrive(file_id, filename, _credentials)
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) 
            stt_col = df.columns[0]
            df[stt_col] = df[stt_col].astype(str).str.strip() 
            return df
        except Exception as e: st.error(f"Lỗi đọc file: {e}")
    return None

def get_or_create_drive_folder(parent_id, folder_name, _credentials):
    """Tìm hoặc tạo folder con trên Drive."""
    service = build('drive', 'v3', credentials=_credentials)
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    if items: return items[0]['id']
    file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
    return service.files().create(body=file_metadata, fields='id').execute().get('id')

def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, _credentials):
    """Cập nhật 'X' vào checklist và lưu ảnh bằng chứng vào folder buổi học (NEW_DATA)."""
    df = st.session_state[CHECKLIST_SESSION_KEY]
    stt_col = df.columns[0]
    row_index = df[df[stt_col] == stt_match].index
    
    if not row_index.empty:
        df.loc[row_index[0], session_name] = 'X'
        st.session_state[CHECKLIST_SESSION_KEY] = df
        
        session_folder_name = session_name.replace("Buổi ", "B")
        target_folder_id = get_or_create_drive_folder(GDRIVE_NEW_DATA_FOLDER_ID, session_folder_name, _credentials)
        
        if target_folder_id:
            drive_filename = f"{session_folder_name}_{stt_match}.jpg"
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.open(io.BytesIO(image_bytes)).convert('RGB').save(tmp.name, format='JPEG')
                upload_to_gdrive_real(tmp.name, target_folder_id, drive_filename, _credentials)
            os.remove(tmp.name)
        return True
    return False

# --- 3. GIAO DIỆN CHÍNH ---

def main_app(credentials):
    if 'camera_input_key' not in st.session_state:
        st.session_state['camera_input_key'] = 0

    from config import download_dataset_folder_real
    download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials)
    
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.session_state[CHECKLIST_SESSION_KEY] = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)

    df = st.session_state[CHECKLIST_SESSION_KEY]
    st.subheader("📋 Danh sách điểm danh")
    st.dataframe(df)

    attendance_cols = [col for col in df.columns if "Buổi" in col]
    selected_session = st.selectbox("Chọn Buổi học", ["--- Vui lòng chọn ---"] + attendance_cols)

    if selected_session != "--- Vui lòng chọn ---":
        captured_file = st.camera_input("Chụp ảnh", key=f"c_{st.session_state['camera_input_key']}")
        
        if captured_file:
            image_bytes = captured_file.getvalue()
            with st.spinner('Đang xử lý...'):
                proc_img, face_detected, num_faces, temp_path = robust_detect_and_crop_face(image_bytes)
                
                if face_detected and num_faces == 1:
                    stt_match, dist = verify_face_against_dataset(temp_path, DATASET_FOLDER)
                    
                    if stt_match:
                        st.success(f"✅ Nhận diện thành công: STT {stt_match}")
                        update_checklist_and_save_new_data(stt_match, selected_session, image_bytes, credentials)
                        st.rerun()
                    
                    else:
                        st.warning("⚠️ Không tìm thấy bạn trong Dataset. Vui lòng chọn tên để đăng ký.")
                        user_options = df.apply(lambda x: f"{x[df.columns[0]]} - {x[df.columns[1]]}", axis=1).tolist()
                        selected_user = st.selectbox("Tên của bạn là:", ["--- Chọn tên ---"] + user_options)
                        
                        if selected_user != "--- Chọn tên ---":
                            chosen_stt = selected_user.split(" - ")[0]
                            if st.button(f"Xác nhận: Tôi là STT {chosen_stt}"):
                                # A. LƯU VÀO DATASET LOCAL (Để nhận diện được ngay lập tức)
                                if not os.path.exists(DATASET_FOLDER):
                                    os.makedirs(DATASET_FOLDER)
                                
                                local_path = os.path.join(DATASET_FOLDER, f"{chosen_stt}.jpg")
                                img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                                img_pil.save(local_path, format='JPEG')
                                
                                # B. XÓA CACHE DEEPFACE (QUAN TRỌNG): Để DeepFace quét lại ảnh mới
                                pkl_path = os.path.join(DATASET_FOLDER, "representations_arcface.pkl")
                                if os.path.exists(pkl_path):
                                    os.remove(pkl_path)
                                
                                # C. UPLOAD LÊN DRIVE (Để lưu trữ vĩnh viễn)
                                upload_to_gdrive_real(local_path, GDRIVE_DATASET_FOLDER_ID, f"{chosen_stt}.jpg", credentials)
                                
                                # D. ĐIỂM DANH
                                update_checklist_and_save_new_data(chosen_stt, selected_session, image_bytes, credentials)
                                
                                st.success(f"Đã đăng ký khuôn mặt cho STT {chosen_stt} và điểm danh!")
                                time.sleep(1)
                                st.session_state['camera_input_key'] += 1
                                st.rerun()
                elif num_faces > 1: st.error("Chỉ chụp 1 người.")
                else: st.error("Không tìm thấy khuôn mặt.")
            if temp_path and os.path.exists(temp_path): os.remove(temp_path)
