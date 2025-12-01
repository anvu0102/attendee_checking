# check.py (Phần logic đã được chỉnh sửa)
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
Đã bổ sung: Checkbox để điều khiển việc hiển thị ảnh đã cắt và ảnh dataset/không khớp VÀ
            Checkbox 'Auto Check' để tự động reset camera sau khi xử lý.
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

# Import hằng số và hàm từ config.py
from config import (
    HAAR_CASCADE_URL, CASCADE_FILENAME, 
    DATASET_FOLDER, CHECKLIST_FILENAME, CHECKLIST_SESSION_KEY, 
    DETECTOR_BACKEND, GDRIVE_CHECKLIST_ID, GDRIVE_NEW_DATA_FOLDER_ID,
    download_file_from_gdrive, upload_to_gdrive_real, list_files_in_gdrive_folder
)

# ... (Các hàm xử lý khác như load_face_cascade, detect_and_draw_face, 
# verify_face_against_dataset, load_checklist, get_next_new_data_stt, 
# check_drive_file_existence, get_or_create_drive_folder, 
# load_dataset_image, update_checklist_and_save_new_data, update_checklist_display 
# giữ nguyên) ...


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    
    # === KHỞI TẠO KEY SESSION STATE ===
    # Khởi tạo key cho camera input nếu chưa có
    if 'camera_input_key' not in st.session_state:
        st.session_state['camera_input_key'] = 0
    # =================================

    # 1. Tải Dataset & Checklist
    from config import GDRIVE_DATASET_FOLDER_ID, GDRIVE_CHECKLIST_ID
    from config import download_dataset_folder_real
    
    # Tải Folder Dataset (REAL)
    dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials) 
    
    # === LOGIC: Tải từ Drive chỉ khi chưa có trong Session State ===
    if CHECKLIST_SESSION_KEY not in st.session_state:
        # Tải Checklist (XLSX) từ Drive
        checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)

        if checklist_df is not None:
            # Lần đầu tiên: Lưu DataFrame vào Session State
            st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
        else:
            st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
            return

    # Lấy DataFrame từ Session State (Sẽ giữ nguyên sau rerun)
    checklist_df = st.session_state[CHECKLIST_SESSION_KEY]
    # ===================================================================
        
    st.markdown("---")

    # Khai báo Placeholder cho checklist
    checklist_placeholder = st.empty()
    
    st.markdown("---") # Thêm vạch phân cách sau Placeholder

    if not dataset_ready:
         st.warning("⚠️ Lỗi tải Dataset Folder. Vui lòng kiểm tra ID Drive Folder và quyền truy cập.")
         return
         
    # Kiểm tra checklist_df (Lấy từ Session State)
    if checklist_df is None:
         st.warning("⚠️ Checklist hiện tại không hợp lệ (Kiểm tra lỗi tải lần đầu).")
         return

    st.info(f"Checklist có {len(checklist_df)} người.")

    # 2. Chọn Buổi Học (Dropdown)
    attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

    if not attendance_cols:
         st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
         return

    # --- THAY ĐỔI: Thêm một tùy chọn mặc định không phải là buổi học ---
    display_options = ["--- Vui lòng chọn buổi ---"] + attendance_cols
    
    selected_session_display = st.selectbox(
        "Chọn Buổi điểm danh", 
        display_options, 
        index=0, 
        help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
    )
    
    # Xác định buổi học thực sự được chọn
    selected_session = selected_session_display if selected_session_display != "--- Vui lòng chọn buổi ---" else None

    # --- BỔ SUNG: CHECKBOX HIỂN THỊ ẢNH DEBUG ---
    col_debug, col_auto = st.columns([0.7, 0.3])
    
    with col_debug:
        show_debug_images = st.checkbox(
            "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
            value=True, 
            help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset."
        )
        
    # --- BỔ SUNG: CHECKBOX AUTO CHECK MỚI ---
    with col_auto:
        # Sử dụng session state để lưu trạng thái Auto Check
        if 'auto_check_enabled' not in st.session_state:
            st.session_state['auto_check_enabled'] = False
            
        auto_check = st.checkbox(
            "Auto Check",
            value=st.session_state['auto_check_enabled'],
            key='auto_check_checkbox', # Sử dụng key để kiểm soát
            help="Tự động reset camera sau khi xử lý ảnh để điểm danh liên tục."
        )
        # Cập nhật session state
        st.session_state['auto_check_enabled'] = auto_check

    # ---------------------------------------------

    st.markdown("---")

    # 3. Chụp Ảnh và Xử Lý
    # --- THAY ĐỔI: Chỉ hiển thị camera input nếu đã chọn buổi ---
    if selected_session:
        
        # --- THÊM KEY VÀO CAMERA INPUT ---
        captured_file = st.camera_input(
            "Chụp ảnh điểm danh", 
            key=f"camera_input_{st.session_state['camera_input_key']}" 
        )
        # ----------------------------------
        
        # Tạo placeholder cho kết quả (để có thể xóa sau 5s)
        result_placeholder = st.empty()

        if captured_file is not None:
            
            # Lấy bytes của ảnh GỐC
            image_bytes_original = captured_file.getvalue() 
            
            stt_match = None
            distance = None
            TEMP_IMAGE_PATH = None
            
            # Khởi tạo cờ cho logic tự động rerun
            should_auto_rerun = False 

            with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
                
                # --- THỰC HIỆN PHÁT HIỆN VÀ TRẢ VỀ TỌA ĐỘ KHUÔN MẶT ---
                processed_image_np, image_original_bgr, face_detected, num_faces, faces = detect_and_draw_face(image_bytes_original, face_cascade)
                processed_image = Image.fromarray(processed_image_np)
                
                # Kiểm tra chỉ có 1 khuôn mặt và tiến hành cắt
                if face_detected and num_faces == 1:
                    # LẤY TỌA ĐỘ KHUÔN MẶT ĐẦU TIÊN
                    (x, y, w, h) = faces[0]
                    
                    # TĂNG KÍCH THƯỚC KHUNG (Padding 20%)
                    padding = int(0.2 * w)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image_original_bgr.shape[1], x + w + padding)
                    y2 = min(image_original_bgr.shape[0], y + h + padding)

                    # CẮT ẢNH KHUÔN MẶT
                    cropped_face_bgr = image_original_bgr[y1:y2, x1:x2]
                    
                    # LƯU ẢNH KHUÔN MẶT ĐÃ CẮT VÀO FILE TẠM cho DeepFace so khớp
                    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    TEMP_IMAGE_PATH = temp_file.name
                    temp_file.close() 
                    
                    cv2.imwrite(TEMP_IMAGE_PATH, cropped_face_bgr)
                    
                    # Thực hiện so khớp DeepFace trên ảnh đã cắt
                    stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)
                
                # --- End If face_detected and num_faces == 1 ---
                
            # HIỂN THỊ KẾT QUẢ TRONG PLACEHOLDER
            with result_placeholder.container():
                st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
                st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

                st.markdown("---")
                st.subheader("💡 Kết quả Điểm danh")
                
                if stt_match and distance is not None: # Đảm bảo cả stt_match và distance đều có giá trị
                    st.balloons()
                    st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
                    
                    # --- BỔ SUNG HIỂN THỊ ẢNH ĐÃ CẮT VÀ ẢNH DATASET TRÙNG KHỚP (CÓ ĐIỀU KIỆN) ---
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
                    # ----------------------------------------------------------------------------
                    
                    st.markdown(f"""
                    * **STT trùng khớp:** **{stt_match}**
                    * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
                    """)
                    
                    # Cập nhật checklist VÀ LƯU ẢNH GỐC THÀNH CÔNG
                    updated = update_checklist_and_save_new_data(stt_match, selected_session, image_bytes_original, credentials)
                    
                    # --- ĐÁNH DẤU CẦN TỰ ĐỘNG RERUN ---
                    should_auto_rerun = auto_check 

                elif face_detected and num_faces == 1:
                    st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
                    
                    # --- BỔ SUNG HIỂN THỊ ẢNH ĐÃ CẮT (CÓ ĐIỀU KIỆN) ---
                    if show_debug_images: 
                        if TEMP_IMAGE_PATH:
                            st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=False)
                    # ----------------------------------------------------
                    
                    # Lưu ảnh gốc (truyền image_bytes_original)
                    update_checklist_and_save_new_data(None, selected_session, image_bytes_original, credentials) 
                    
                    # --- ĐÁNH DẤU CẦN TỰ ĐỘNG RERUN ---
                    should_auto_rerun = auto_check

                elif face_detected and num_faces > 1:
                    st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")
                    
                    # --- ĐÁNH DẤU CẦN TỰ ĐỘNG RERUN ---
                    should_auto_rerun = auto_check

                else:
                    st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
                    st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")
                    
                    # --- ĐÁNH DẤU CẦN TỰ ĐỘNG RERUN ---
                    should_auto_rerun = auto_check

                # --- LOGIC TỰ ĐỘNG CLEAR VÀ RERUN (CHỈ KHI AUTO CHECK BẬT) ---
                if should_auto_rerun:
                    time.sleep(5) # Đợi 5 giây (Theo yêu cầu của người dùng)
                    
                    # Xóa file tạm sau khi đã hiển thị xong (trước khi rerun)
                    if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                        os.remove(TEMP_IMAGE_PATH)
                        
                    # Tăng giá trị key để buộc Streamlit reset widget st.camera_input
                    st.session_state['camera_input_key'] += 1 
                    st.rerun() # Buộc rerun
                    # --------------------------------------
                    
            # --- Vị trí XÓA file tạm mới: Xóa file tạm nếu không vào khối logic tự động clear ---
            if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
            # ---------------------------------------------------------------------------------------
            
    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU HOẶC SAU KHI RERUN
    if CHECKLIST_SESSION_KEY in st.session_state:
        # Nếu có cập nhật (từ khối IF stt_match) VÀ KHÔNG RERUN TỰ ĐỘNG, cập nhật hiển thị ngay
        # Hoặc chỉ đơn giản là hiển thị lại trạng thái hiện tại
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
