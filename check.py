# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
ĐÃ CẬP NHẬT: Loại bỏ st.camera_input, thay thế bằng streamlit_webrtc để xử lý luồng video trực tiếp 
(Loại bỏ nhu cầu nhấn nút 'Take Photo').
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
# --- THƯ VIỆN BỔ SUNG CHO WEBRTC ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Cần cho việc chuyển đổi frame
# ------------------------------------

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


# ----------------------------------------------------------------------
#                             CÁC HÀM XỬ LÝ (GIỮ NGUYÊN)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải Haar Cascade...")
def load_face_cascade(url, filename):
    """ Tải Haar Cascade cho OpenCV. """
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


def detect_and_draw_face(image_np_bgr, cascade):
    """ 
    Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh (BGR). 
    Trả về: ảnh có khung (RGB), cờ phát hiện, số lượng khuôn mặt, TỌA ĐỘ (x,y,w,h).
    """
    
    image_original_bgr = image_np_bgr.copy()
    image_bgr_with_frame = image_original_bgr.copy()
    
    gray = cv2.cvtColor(image_original_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB)

    # TRẢ VỀ: (ảnh có khung (RGB), cờ phát hiện, số lượng khuôn mặt, TỌA ĐỘ KHUÔN MẶT)
    return processed_image_rgb, len(faces) > 0, len(faces), faces


def verify_face_against_dataset(target_image_path, dataset_folder):
    """ 
    Sử dụng DeepFace để so sánh ảnh đầu vào (ĐÃ CẮT) với dataset. 
    """
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
            
            if pd.notna(distance):
                return stt_match, float(distance)
            else:
                return None, None
                
        return None, None
    except Exception as e:
        if "Face could not be detected" not in str(e):
            # Chỉ hiển thị lỗi DeepFace nếu không phải lỗi không phát hiện
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


def load_checklist(file_id, filename, _credentials):
    """ Tải checklist XLSX và đọc thành DataFrame. """
    download_file_from_gdrive(file_id, filename, _credentials)
        
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) 
            stt_col = df.columns[0]
            df[stt_col] = df[stt_col].astype(str).str.strip() 
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

# --- HÀM TÌM SỐ THỨ TỰ LỚN NHẤT VÀ KIỂM TRA TỒN TẠI (GIỮ NGUYÊN) ---
def get_next_new_data_stt(_credentials):
    # ... (giữ nguyên hàm)
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
    # ... (giữ nguyên hàm)
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
    # ... (giữ nguyên hàm)
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
        
def load_dataset_image(stt_match, dataset_folder):
    # ... (giữ nguyên hàm)
    pattern_simple = re.compile(rf'^{stt_match}\.jpe?g$', re.IGNORECASE)
    pattern_complex = re.compile(rf'^{stt_match}_.*\.jpe?g$', re.IGNORECASE)
    
    if os.path.isdir(dataset_folder):
        for filename in os.listdir(dataset_folder):
            if pattern_simple.match(filename):
                return os.path.join(dataset_folder, filename)
            if pattern_complex.match(filename):
                return os.path.join(dataset_folder, filename)
    return None

        
# --- HÀM CẬP NHẬT CHECKLIST VÀ LƯU ẢNH (CHỈNH SỬA ĐỂ NHẬN ẢNH BGR) ---
def update_checklist_and_save_new_data(stt_match, session_name, image_np_bgr, _credentials):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới lên Drive.
    Lưu ý: image_np_bgr là mảng numpy của ảnh GỐC (BGR).
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return False 

    df = st.session_state[CHECKLIST_SESSION_KEY]
    updated = False 
    
    # Chuyển đổi NumPy BGR sang PIL RGB để lưu vào BytesIO/File
    image_to_save_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_to_save_rgb)
    
    # Chuẩn bị BytesIO cho ảnh gốc
    output = io.BytesIO()
    image_pil.save(output, format='JPEG')
    image_bytes = output.getvalue()


    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        try:
            stt_col = df.columns[0] 
            row_index = df[df[stt_col] == stt_match].index
            
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
                        # Lưu ảnh từ bytes (image_bytes - LÚC NÀY LÀ ẢNH GỐC) vào file tạm
                        image_pil.save(TEMP_UPLOAD_PATH, format='JPEG')
                        
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
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp) 
    else: 
        st.warning("⚠️ Đang lưu ảnh vào folder dữ liệu mới...")
        next_counter = get_next_new_data_stt(_credentials)
        session_num = session_name.replace("Buổi ", "")
        drive_filename = f"B{session_num}_{next_counter}.jpg" 
        
        temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_UPLOAD_PATH = temp_file_for_upload.name
        temp_file_for_upload.close()
        
        try:
            image_pil.save(TEMP_UPLOAD_PATH, format='JPEG')
            upload_to_gdrive_real(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, _credentials)
            st.info(f"🖼️ Đã lưu ảnh không khớp vào folder chung: {drive_filename}")

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)
                
    return updated 


# --- HÀM XỬ LÝ FRAME SỐNG (LIVE FRAME PROCESSING LOGIC) ---
def process_live_frame(image_np_bgr, selected_session, credentials, show_debug_images):
    """
    Hàm xử lý DeepFace cho một khung hình duy nhất,
    cập nhật checklist và hiển thị kết quả.
    image_np_bgr: Mảng NumPy BGR của khung hình hiện tại.
    """
    stt_match = None
    distance = None
    TEMP_IMAGE_PATH = None
    
    # --- 1. PHÁT HIỆN VÀ CẮT KHUÔN MẶT ---
    # Ảnh BGR gốc (chỉ dùng để cắt)
    image_original_bgr = image_np_bgr.copy() 
    
    processed_image_rgb, face_detected, num_faces, faces = detect_and_draw_face(image_original_bgr, face_cascade)
    
    if face_detected and num_faces == 1:
        (x, y, w, h) = faces[0]
        padding = int(0.2 * w)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_original_bgr.shape[1], x + w + padding)
        y2 = min(image_original_bgr.shape[0], y + h + padding)

        cropped_face_bgr = image_original_bgr[y1:y2, x1:x2]
        
        # LƯU ẢNH KHUÔN MẶT ĐÃ CẮT VÀO FILE TẠM
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_IMAGE_PATH = temp_file.name
        temp_file.close() 
        cv2.imwrite(TEMP_IMAGE_PATH, cropped_face_bgr)
        
        # --- 2. SO KHỚP DEEPFACE ---
        stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)
    
    # --- 3. HIỂN THỊ VÀ CẬP NHẬT KẾT QUẢ ---
    
    # Sử dụng st.container() để chứa kết quả xử lý
    with st.container():
        st.subheader("🖼️ Khuôn mặt đã phát hiện")
        st.image(processed_image_rgb, caption="Khuôn mặt được đánh dấu trong khung hình.", width='stretch')
        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if stt_match and distance is not None: 
            st.balloons()
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            
            # --- Hiển thị ảnh debug ---
            if show_debug_images: 
                dataset_image_path = load_dataset_image(stt_match, DATASET_FOLDER)
                col1, col2 = st.columns(2)
                with col1:
                    if TEMP_IMAGE_PATH:
                        st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", width='stretch')
                with col2:
                    if dataset_image_path:
                        st.image(dataset_image_path, caption=f"Dataset (STT: {stt_match})", width='stretch')
            
            st.markdown(f"""
            * **STT trùng khớp:** **{stt_match}**
            * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
            """)
            
            # Cập nhật checklist VÀ LƯU ẢNH GỐC THÀNH CÔNG
            updated = update_checklist_and_save_new_data(stt_match, selected_session, image_original_bgr, credentials)
            
            # Quay lại trạng thái chờ sau 5 giây để tránh xử lý lặp lại ngay lập tức
            if updated:
                st.info("Đã cập nhật checklist thành công. Tự động reset sau 5 giây.")
                time.sleep(5) 
            
        elif face_detected and num_faces == 1:
            st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
            if show_debug_images and TEMP_IMAGE_PATH: 
                st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", width='content')
            
            # Lưu ảnh gốc không khớp
            update_checklist_and_save_new_data(None, selected_session, image_original_bgr, credentials) 
            st.info("Đã lưu ảnh không khớp. Tự động reset sau 5 giây.")
            time.sleep(5)

        elif face_detected and num_faces > 1:
            st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")
            st.info("Tự động reset sau 5 giây.")
            time.sleep(5)
            
        else:
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.info("Tự động reset sau 5 giây.")
            time.sleep(5)
            
    # Xóa file tạm sau khi đã xử lý
    if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)
        
    # Buộc Streamlit rerun để xóa giao diện kết quả và quay lại trạng thái chờ
    st.rerun()

# --- HÀM CẬP NHẬT PLACEHOLDER CHECKLIST (GIỮ NGUYÊN) ---
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


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    
    # === KHỞI TẠO KEY SESSION STATE ===
    # Sử dụng 'processing_triggered' để theo dõi trạng thái kích hoạt xử lý
    if 'processing_triggered' not in st.session_state:
        st.session_state['processing_triggered'] = False
    # Sử dụng 'webrtc_key' để reset widget
    if 'webrtc_key' not in st.session_state:
        st.session_state['webrtc_key'] = 0
    # =================================

    # 1. Tải Dataset & Checklist
    from config import GDRIVE_DATASET_FOLDER_ID, GDRIVE_CHECKLIST_ID
    from config import download_dataset_folder_real
    
    dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials) 
    
    if CHECKLIST_SESSION_KEY not in st.session_state:
        checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)
        if checklist_df is not None:
            st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
        else:
            st.warning("⚠️ Lỗi tải hoặc đọc file Checklist.")
            return

    checklist_df = st.session_state[CHECKLIST_SESSION_KEY]
        
    st.markdown("---")

    checklist_placeholder = st.empty()
    
    st.markdown("---") 

    if not dataset_ready:
         st.warning("⚠️ Lỗi tải Dataset Folder.")
         return
         
    if checklist_df is None:
         st.warning("⚠️ Checklist hiện tại không hợp lệ.")
         return

    st.info(f"Checklist có {len(checklist_df)} người.")

    # 2. Chọn Buổi Học (Dropdown)
    attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

    if not attendance_cols:
         st.error("Không tìm thấy cột 'Buổi' trong checklist.")
         return

    display_options = ["--- Vui lòng chọn buổi ---"] + attendance_cols
    
    selected_session_display = st.selectbox(
        "Chọn Buổi điểm danh", 
        display_options, 
        index=0, 
        help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
    )
    
    selected_session = selected_session_display if selected_session_display != "--- Vui lòng chọn buổi ---" else None

    # --- CHECKBOX HIỂN THỊ ẢNH DEBUG ---
    show_debug_images = st.checkbox(
        "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
        value=True, 
        help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset (khi điểm danh thành công) hoặc ảnh đã cắt (khi không khớp)."
    )

    st.markdown("---")

    # 3. KÍCH HOẠT WEBRTC VÀ XỬ LÝ KHUNG HÌNH
    if selected_session:
        
        col_video, col_trigger = st.columns([2, 1])

        # --- VIDEO STREAM ---
        with col_video:
            st.subheader("📹 Luồng Video Trực tiếp")
            # Sử dụng key để có thể reset widget
            webrtc_ctx = webrtc_streamer(
                key=f"webrtc_{st.session_state['webrtc_key']}", 
                video_transformer_factory=None, 
                media_stream_constraints={"video": True, "audio": False},
                async_transform=False # Xử lý đồng bộ (mặc dù ta không dùng transformer)
            )

        # --- TRIGGER BUTTON ---
        with col_trigger:
            st.subheader("Kích hoạt")
            # Button để kích hoạt việc lấy khung hình và xử lý
            if st.button("🔴 Kích hoạt Xử lý/Điểm danh", help="Nhấn để lấy khung hình hiện tại và thực hiện nhận diện.", disabled=not (webrtc_ctx and webrtc_ctx.state.playing)):
                st.session_state['processing_triggered'] = True
                # Buộc Streamlit rerun để thực thi logic xử lý bên dưới
                st.rerun()

        # --- LOGIC XỬ LÝ SAU KHI KÍCH HOẠT ---
        if st.session_state['processing_triggered'] and webrtc_ctx and webrtc_ctx.state.playing:
            
            # Reset cờ kích hoạt
            st.session_state['processing_triggered'] = False
            
            latest_frame = webrtc_ctx.get_last_frame()
            
            if latest_frame:
                with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
                    # Chuyển đổi khung hình AV (RGB) sang mảng NumPy BGR cho OpenCV
                    image_np_rgb = latest_frame.to_ndarray(format="rgb24")
                    image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

                    # --- GỌI HÀM XỬ LÝ FRAME SỐNG ---
                    process_live_frame(image_np_bgr, selected_session, credentials, show_debug_images)
                    
                    # Nếu process_live_frame gọi rerun, code dưới đây sẽ không chạy
            else:
                st.warning("⚠️ Không thể lấy khung hình. Đảm bảo camera đã hoạt động.")
                time.sleep(2)
                st.rerun()
                
    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU HOẶC SAU KHI RERUN
    if CHECKLIST_SESSION_KEY in st.session_state:
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
