# check.py
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

# Import hằng số và hàm từ config.py
from config import (
    DATASET_FOLDER, CHECKLIST_FILENAME, CHECKLIST_SESSION_KEY, 
    DETECTOR_BACKEND, GDRIVE_CHECKLIST_ID, GDRIVE_DATASET_FOLDER_ID,
    GDRIVE_NEW_DATA_FOLDER_ID,
    download_file_from_gdrive, upload_to_gdrive_real, list_files_in_gdrive_folder
)

# --- CÁC HÀM XỬ LÝ CƠ BẢN ---

def robust_detect_and_crop_face(image_bytes):
    """Sử dụng DeepFace để phát hiện và cắt khuôn mặt."""
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    image_bgr_with_frame = image_original_bgr.copy()
    
    face_detected = False
    num_faces = 0
    temp_cropped_path = None
    
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
            
    except Exception:
        pass
        
    processed_image_rgb = cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB)
    return processed_image_rgb, face_detected, num_faces, temp_cropped_path

def verify_face_against_dataset(target_image_path, dataset_folder):
    """So khớp khuôn mặt với database hiện có."""
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
            stt_match = os.path.splitext(os.path.basename(identity_path))[0].split('_')[0]
            distance = best_match['ArcFace_cosine'] 
            return stt_match, float(distance)
        return None, None
    except Exception:
        return None, None

def load_checklist(file_id, filename, _credentials):
    """Tải và chuẩn hóa dữ liệu checklist."""
    download_file_from_gdrive(file_id, filename, _credentials)
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) 
            stt_col = df.columns[0]
            df[stt_col] = df[stt_col].astype(str).str.strip() 
            return df
        except Exception as e:
            st.error(f"Lỗi đọc checklist: {e}")
    return None

def update_checklist_display(checklist_placeholder, current_df):
    """Hiển thị bảng checklist lên giao diện."""
    with checklist_placeholder.container():
        st.subheader("📋 Trạng thái Checklist Hiện tại")
        st.dataframe(current_df)

def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, _credentials):
    """Đánh dấu điểm danh và lưu ảnh bằng chứng vào folder buổi học."""
    df = st.session_state[CHECKLIST_SESSION_KEY]
    stt_col = df.columns[0]
    row_index = df[df[stt_col] == stt_match].index
    
    if not row_index.empty:
        # Đánh dấu X
        df.loc[row_index[0], session_name] = 'X'
        st.session_state[CHECKLIST_SESSION_KEY] = df
        
        # Tạo folder buổi học trên Drive và upload ảnh bằng chứng
        session_folder_name = session_name.replace("Buổi ", "B")
        from check import get_or_create_drive_folder # Đảm bảo hàm này tồn tại
        target_folder_id = get_or_create_drive_folder(GDRIVE_NEW_DATA_FOLDER_ID, session_folder_name, _credentials)
        
        if target_folder_id:
            drive_filename = f"{session_folder_name}_{stt_match}.jpg"
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.open(io.BytesIO(image_bytes)).convert('RGB').save(tmp.name, format='JPEG')
                upload_to_gdrive_real(tmp.name, target_folder_id, drive_filename, _credentials)
            os.remove(tmp.name)
        return True
    return False

def get_or_create_drive_folder(parent_id, folder_name, _credentials):
    """Tìm hoặc tạo folder trên Drive."""
    service = build('drive', 'v3', credentials=_credentials)
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
    file = service.files().create(body=file_metadata, fields='id').execute()
    return file.get('id')

# --- GIAO DIỆN CHÍNH ---

def main_app(credentials):
    if 'camera_input_key' not in st.session_state:
        st.session_state['camera_input_key'] = 0

    from config import download_dataset_folder_real
    download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials)
    
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.session_state[CHECKLIST_SESSION_KEY] = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)

    df = st.session_state[CHECKLIST_SESSION_KEY]
    checklist_placeholder = st.empty()
    update_checklist_display(checklist_placeholder, df)

    # Chọn buổi học
    attendance_cols = [col for col in df.columns if "Buổi" in col]
    selected_session = st.selectbox("Chọn Buổi điểm danh", ["--- Vui lòng chọn buổi ---"] + attendance_cols)

    if selected_session != "--- Vui lòng chọn buổi ---":
        captured_file = st.camera_input("Chụp ảnh điểm danh", key=f"cam_{st.session_state['camera_input_key']}")
        
        if captured_file:
            image_bytes = captured_file.getvalue()
            with st.spinner('Đang nhận diện...'):
                proc_img, face_detected, num_faces, temp_path = robust_detect_and_crop_face(image_bytes)
                
                if face_detected and num_faces == 1:
                    stt_match, dist = verify_face_against_dataset(temp_path, DATASET_FOLDER)
                    
                    # TRƯỜNG HỢP 1: ĐÃ CÓ DỮ LIỆU
                    if stt_match:
                        st.success(f"✅ Nhận diện thành công STT: {stt_match}")
                        if update_checklist_and_save_new_data(stt_match, selected_session, image_bytes, credentials):
                            st.balloons()
                            time.sleep(2)
                            st.session_state['camera_input_key'] += 1
                            st.rerun()
                    
                    # TRƯỜNG HỢP 2: NGƯỜI MỚI - CHỌN TỪ DROPDOWN
                    else:
                        st.warning("⚠️ Không tìm thấy khuôn mặt trong hệ thống. Vui lòng chọn tên để đăng ký.")
                        # Tạo danh sách "STT - Họ Tên"
                        user_options = df.apply(lambda x: f"{x[df.columns[0]]} - {x[df.columns[1]]}", axis=1).tolist()
                        selected_user = st.selectbox("Bạn là ai trong danh sách này?", ["--- Chọn tên ---"] + user_options)
                        
                        if selected_user != "--- Chọn tên ---":
                            chosen_stt = selected_user.split(" - ")[0]
                            if st.button(f"Xác nhận đăng ký khuôn mặt cho STT {chosen_stt}"):
                                # 1. Upload ảnh vào DATASET để học (Đặt tên là STT.jpg)
                                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_ds:
                                    Image.open(io.BytesIO(image_bytes)).convert('RGB').save(tmp_ds.name, format='JPEG')
                                    upload_to_gdrive_real(tmp_ds.name, GDRIVE_DATASET_FOLDER_ID, f"{chosen_stt}.jpg", credentials)
                                
                                # 2. Tiến hành điểm danh
                                update_checklist_and_save_new_data(chosen_stt, selected_session, image_bytes, credentials)
                                st.success("Đã lưu dữ liệu và điểm danh thành công!")
                                os.remove(tmp_ds.name)
                                time.sleep(2)
                                st.session_state['camera_input_key'] += 1
                                st.rerun()
                elif num_faces > 1:
                    st.error("Chỉ cho phép 1 người trong khung hình.")
                else:
                    st.error("Không tìm thấy khuôn mặt.")

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
