# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
Đã chuyển đổi sang sử dụng streamlit-webrtc để hỗ trợ Real-time Face Detection và Auto-Capture.
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

# BỔ SUNG THƯ VIỆN CHO REAL-TIME VIDEO STREAM
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av # Cần thiết cho việc xử lý khung hình video

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
#                             CÁC HÀM XỬ LÝ
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


# BỎ HÀM detect_and_draw_face CŨ VÌ LOGIC ĐƯỢC CHUYỂN VÀO CLASS FaceDetectionProcessor

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
                st.error("❌ DeepFace không trả về độ tương đồng (distance) hợp lệ.")
                return None, None
                
        return None, None
    except Exception as e:
        if "Face could not be detected" in str(e):
             # Lỗi này có thể xảy ra do ảnh cắt chất lượng kém
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. (Kiểm tra chất lượng ảnh)")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


def load_checklist(file_id, filename, _credentials):
    """ Tải checklist XLSX và đọc thành DataFrame từ Drive. """
    
    download_file_from_gdrive(file_id, filename, _credentials)
        
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) 
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

def get_next_new_data_stt(_credentials):
    """ Tìm số thứ tự lớn nhất trong folder NEW_DATA_FOLDER_ID trên Drive. """
    
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
    """ Kiểm tra xem file có tên filename đã tồn tại trong folder_id trên Drive hay chưa. """
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
    """ Tìm ID của folder con trong parent_id. Nếu chưa tồn tại, tạo mới. """
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
    """ Tìm và trả về đường dẫn của ảnh dataset tương ứng với STT match đầu tiên. """
    pattern_simple = re.compile(rf'^{stt_match}\.jpe?g$', re.IGNORECASE)
    pattern_complex = re.compile(rf'^{stt_match}_.*\.jpe?g$', re.IGNORECASE)
    
    if os.path.isdir(dataset_folder):
        for filename in os.listdir(dataset_folder):
            
            if pattern_simple.match(filename):
                return os.path.join(dataset_folder, filename)
                
            if pattern_complex.match(filename):
                return os.path.join(dataset_folder, filename)
                
    return None
        
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
                        # Lưu ảnh từ bytes (image_bytes - LÚC NÀY LÀ ẢNH GỐC) vào file tạm
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

                
                # Kiểm tra nếu chưa điểm danh thì mới cập nhật (NGĂN TRÙNG LẶP)
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
#                             CLASS XỬ LÝ VIDEO REAL-TIME
# ----------------------------------------------------------------------

class FaceDetectionProcessor(VideoProcessorBase):
    """
    Xử lý từng khung hình để phát hiện và vẽ khung khuôn mặt. 
    Nếu phát hiện 1 khuôn mặt, lưu khung hình vào Session State để kích hoạt logic DeepFace.
    """
    def __init__(self, face_cascade):
        self.face_cascade = face_cascade
        
    def recv(self, frame):
        """ Nhận một khung hình và trả về khung hình đã xử lý. """
        
        img = frame.to_ndarray(format="bgr24") 
        
        # Sao chép ảnh để vẽ khung
        img_with_frame = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # --- LOGIC TỰ ĐỘNG CHỤP VÀ LƯU VÀO SESSION STATE ---
        # Kiểm tra nếu chưa có ảnh nào đang chờ xử lý và có đúng 1 khuôn mặt
        if len(faces) == 1 and st.session_state.get('processing_frame', False) == False:
            
            # Lưu ảnh gốc (bgr) và tọa độ khuôn mặt vào Session State
            st.session_state['captured_frame'] = img.copy() 
            st.session_state['face_coords'] = faces[0]
            st.session_state['processing_frame'] = True # Đánh dấu đang chờ xử lý
            
            # Vẽ khung màu đỏ để báo hiệu đã chụp/chờ xử lý
            (x, y, w, h) = faces[0]
            cv2.rectangle(img_with_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Sau khi lưu vào Session State, Streamlit sẽ tự động rerun khi luồng video trả về.
            # Không cần gọi st.rerun() trực tiếp từ đây.
            
        else:
            # Vẽ khung màu xanh lá nếu có khuôn mặt
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            
        return av.VideoFrame.from_ndarray(img_with_frame, format="bgr24")

# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    
    # === KHỞI TẠO KEY SESSION STATE ===
    # Khởi tạo key cho camera input nếu chưa có (Dùng cho logic DeepFace)
    if 'processing_frame' not in st.session_state:
        st.session_state['processing_frame'] = False # Cờ kiểm tra ảnh đang chờ xử lý
    if 'captured_frame' not in st.session_state:
        st.session_state['captured_frame'] = None
    if 'face_coords' not in st.session_state:
        st.session_state['face_coords'] = None
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
            st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
            return

    checklist_df = st.session_state[CHECKLIST_SESSION_KEY]
        
    st.markdown("---")

    # Khai báo Placeholder cho checklist
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
    col_debug, col_auto = st.columns(2) 
    
    with col_debug:
        show_debug_images = st.checkbox(
            "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
            value=True, 
            help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset (khi điểm danh thành công) hoặc ảnh đã cắt (khi không khớp)."
        )

    with col_auto: 
        auto_check_enabled = st.checkbox(
            "Tự động clear & tiếp tục (Auto Check)",
            value=False, 
            help="Nếu được bật, sau khi điểm danh thành công, màn hình sẽ tự động clear và chuẩn bị cho lần chụp tiếp theo sau 2 giây."
        )
        st.session_state['auto_check_enabled'] = auto_check_enabled # Lưu cờ vào Session State để VideoProcessor truy cập
    # ---------------------------------------------

    st.markdown("---")

    # 3. KÍCH HOẠT LUỒNG VIDEO & XỬ LÝ ẢNH ĐÃ TỰ ĐỘNG CHỤP
    if selected_session:
        
        st.subheader("🔴 Luồng Video Trực tiếp (Tự động chụp khi phát hiện 1 khuôn mặt)")
        
        # --- STREAMLIT-WEBRTC WIDGET ---
        webrtc_ctx = webrtc_streamer(
            key="webcam_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: FaceDetectionProcessor(face_cascade),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        # ------------------------------
        
        # Sử dụng placeholder để hiển thị kết quả (Nếu có ảnh được chụp)
        result_placeholder = st.empty()

        # --- LOGIC XỬ LÝ HẬU KỲ (DEEPFACE) KHI CÓ KHUNG HÌNH ĐƯỢC CHỤP ---
        # Nếu có khung hình được tự động chụp trong Session State (do VideoProcessor kích hoạt)
        if st.session_state['captured_frame'] is not None and st.session_state.get('processing_frame', False) == True:
            
            # Lấy dữ liệu và dọn dẹp Session State (trừ cờ processing_frame để giữ luồng video tạm nghỉ)
            image_original_bgr = st.session_state.pop('captured_frame')
            faces_coords = [st.session_state.pop('face_coords')]
            
            # Chuyển ảnh BGR về bytes (phù hợp với update_checklist_and_save_new_data)
            _, image_bytes_original = cv2.imencode('.jpg', image_original_bgr)
            image_bytes_original = image_bytes_original.tobytes()
            
            stt_match = None
            distance = None
            TEMP_IMAGE_PATH = None
            face_detected = True
            num_faces = 1
            
            with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
                
                # CHỈ XỬ LÝ TIẾP NẾU CÓ ĐÚNG 1 KHUÔN MẶT ĐÃ ĐƯỢC CHỤP (Đã kiểm tra trong VideoProcessor)
                if num_faces == 1:
                    
                    (x, y, w, h) = faces_coords[0]
                    
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
                
                # --- VẼ ẢNH ĐÃ CHỤP (CÓ KHUNG) ---
                processed_image_rgb = cv2.cvtColor(cv2.rectangle(image_original_bgr.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2), cv2.COLOR_BGR2RGB)
                processed_image = Image.fromarray(processed_image_rgb)
                
            # HIỂN THỊ KẾT QUẢ TRONG PLACEHOLDER
            with result_placeholder.container():
                st.subheader("🖼️ Ảnh đã Tự động Chụp và Nhận diện")
                st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

                st.markdown("---")
                st.subheader("💡 Kết quả Điểm danh")
                
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
                    
                    updated = update_checklist_and_save_new_data(stt_match, selected_session, image_bytes_original, credentials)
                    
                    if updated and CHECKLIST_SESSION_KEY in st.session_state:
                         update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
                    
                    # --- LOGIC TỰ ĐỘNG CLEAR ---
                    if auto_check_enabled: 
                        time.sleep(2) 
                        
                        if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                            os.remove(TEMP_IMAGE_PATH)
                            
                        # QUAN TRỌNG: Gỡ cờ xử lý để VideoProcessor có thể chụp khung hình mới
                        st.session_state['processing_frame'] = False 
                        st.rerun() 
                        return 
                    
                elif face_detected and num_faces == 1:
                    st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
                    
                    if show_debug_images: 
                        if TEMP_IMAGE_PATH:
                            st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=False)
                    
                    update_checklist_and_save_new_data(None, selected_session, image_bytes_original, credentials) 
                
                # KHÔNG BAO GỒM CÁC TRƯỜNG HỢP NHIỀU KHUÔN MẶT/KHÔNG KHUÔN MẶT VÌ ĐÃ LỌC TRONG VIDEO PROCESSOR

            # --- Dọn dẹp file tạm và cờ (Nếu không tự động rerun) ---
            if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
            
            # Gỡ cờ xử lý để VideoProcessor có thể chụp khung hình mới (nếu không auto check)
            st.session_state['processing_frame'] = False 

    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU HOẶC SAU KHI RERUN
    if CHECKLIST_SESSION_KEY in st.session_state:
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
