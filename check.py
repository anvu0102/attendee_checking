# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
ĐÃ CẢI TIẾN: Thay thế Haar Cascade bằng DeepFace.extract_faces (sử dụng DETECTOR_BACKEND) 
để tăng độ ổn định phát hiện khuôn mặt (ví dụ: khi nghiêng đầu).
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
    HAAR_CASCADE_URL, CASCADE_FILENAME, # Giữ lại import nhưng không dùng trong logic phát hiện mới
    DATASET_FOLDER, CHECKLIST_FILENAME, CHECKLIST_SESSION_KEY, 
    DETECTOR_BACKEND, GDRIVE_CHECKLIST_ID, GDRIVE_NEW_DATA_FOLDER_ID,
    download_file_from_gdrive, upload_to_gdrive_real, list_files_in_gdrive_folder
)


# ----------------------------------------------------------------------
#                             CÁC HÀM XỬ LÝ
# ----------------------------------------------------------------------

# --- HÀM MỚI: PHÁT HIỆN, VẼ KHUNG VÀ CẮT ẢNH MẠNH MẼ HƠN ---
def robust_detect_and_crop_face(image_bytes):
    """ 
    Sử dụng DeepFace.extract_faces để phát hiện khuôn mặt, vẽ khung, cắt ảnh và lưu vào file tạm. 
    Trả về: ảnh có khung (RGB), cờ phát hiện, số lượng khuôn mặt, đường dẫn file ảnh đã cắt (tạm thời).
    """
    
    # Đọc ảnh từ bytes
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    image_bgr_with_frame = image_original_bgr.copy() # Bản sao để vẽ khung
    
    face_detected = False
    num_faces = 0
    temp_cropped_path = None
    
    try:
        # Sử dụng DETECTOR_BACKEND (ví dụ: MTCNN/RetinaFace) cho độ ổn định cao
        faces_extracted = DeepFace.extract_faces(
            img_path=image_np, # Truyền numpy array
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=False # Để hàm không ném ValueError khi không tìm thấy
        )
        
        num_faces = len(faces_extracted)
        face_detected = num_faces > 0

        if num_faces == 1:
            # Lấy tọa độ khuôn mặt đầu tiên
            # DeepFace trả về tọa độ (x, y, w, h) trong 'facial_area'
            facial_area = faces_extracted[0]['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # 1. Vẽ khung lên bản sao (Màu Xanh Dương cho 1 khuôn mặt)
            cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 2. Cắt ảnh (có padding)
            padding = int(0.2 * w)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_original_bgr.shape[1], x + w + padding)
            y2 = min(image_original_bgr.shape[0], y + h + padding)

            cropped_face_bgr = image_original_bgr[y1:y2, x1:x2]
            
            # 3. LƯU ẢNH KHUÔN MẶT ĐÃ CẮT VÀO FILE TẠM cho DeepFace so khớp
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_cropped_path = temp_file.name
            temp_file.close() 
            
            cv2.imwrite(temp_cropped_path, cropped_face_bgr)
            
        elif num_faces > 1:
            # Vẫn vẽ khung cho tất cả các khuôn mặt để người dùng thấy lỗi (Màu Đỏ cho > 1 khuôn mặt)
            for face_data in faces_extracted:
                facial_area = face_data['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 

    except Exception as e:
        # st.error(f"❌ Lỗi trong quá trình phát hiện khuôn mặt: {e}")
        pass # Bỏ qua lỗi nhỏ để luồng chính xử lý
        
    processed_image_rgb = cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB)
    
    # TRẢ VỀ: (ảnh có khung (RGB), cờ phát hiện, số lượng khuôn mặt, đường dẫn file đã cắt)
    return processed_image_rgb, face_detected, num_faces, temp_cropped_path


def verify_face_against_dataset(target_image_path, dataset_folder):
    """ 
    Sử dụng DeepFace để so sánh ảnh đầu vào (ĐÃ CẮT) với dataset. 
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
            
            # Lấy STT từ tên file (vd: 1_001.jpg -> 1)
            stt_match = os.path.splitext(os.path.basename(identity_path))[0].split('_')[0]
            distance = best_match['ArcFace_cosine'] 
            
            if pd.notna(distance):
                return stt_match, float(distance)
            else:
                st.error("❌ DeepFace không trả về độ tương đồng (distance) hợp lệ.")
                return None, None
                
        return None, None
    except Exception as e:
        # Chỉ in lỗi DeepFace nếu không phải lỗi không phát hiện
        if "Face could not be detected" in str(e):
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. (Kiểm tra chất lượng ảnh)")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


# BỎ DECORATOR @st.cache_data để buộc tải lại checklist mỗi khi app load
def load_checklist(file_id, filename, _credentials):
    """ 
    Tải checklist XLSX và đọc thành DataFrame. 
    Hàm này **luôn** tải lại file từ Drive để lấy dữ liệu mới nhất.
    """
    
    # 1. Tải file checklist mới nhất từ Drive (ghi đè lên file local nếu có)
    download_file_from_gdrive(file_id, filename, _credentials)
        
    # 2. Đọc file local vừa tải
    if os.path.exists(filename):
        try:
            # ĐỌC FILE XLSX
            df = pd.read_excel(filename) 
            
            # === FIX LỖI PYARROW (ArrowTypeError): Chuẩn hóa cột STT thành STRING ===
            # Giả định cột STT là cột đầu tiên
            stt_col = df.columns[0]
            # Chuyển đổi thành chuỗi và loại bỏ khoảng trắng dư thừa
            df[stt_col] = df[stt_col].astype(str).str.strip() 
            # =======================================================================
            
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

# --- HÀM TÌM SỐ THỨ TỰ LỚN NHẤT TRONG FOLDER NEW DATA ---
def get_next_new_data_stt(_credentials):
    """
    Tìm số thứ tự lớn nhất trong folder NEW_DATA_FOLDER_ID trên Drive
    để đặt tên cho file mới (ví dụ: B1_1.jpg, B1_2.jpg, ...).
    Trả về số thứ tự tiếp theo (integer).
    """
    
    # 1. Lấy danh sách tên file từ Drive
    file_list = list_files_in_gdrive_folder(GDRIVE_NEW_DATA_FOLDER_ID, _credentials)
    
    max_stt = 0
    # Biểu thức chính quy để tìm số sau dấu gạch dưới (ví dụ: BX_123.jpg -> 123)
    # Pattern: [Buổi]<số>_<số>.jpg
    pattern = re.compile(r'B\d+_(\d+)\.jpe?g$', re.IGNORECASE)
    
    for filename in file_list:
        match = pattern.search(filename)
        if match:
            try:
                # Lấy số thứ tự (group 1)
                stt = int(match.group(1))
                if stt > max_stt:
                    max_stt = stt
            except ValueError:
                continue

    # Trả về số thứ tự tiếp theo
    return max_stt + 1

# --- HÀM: KIỂM TRA TÊN FILE TỒN TẠI TRONG FOLDER DRIVE ---
def check_drive_file_existence(folder_id, filename, _credentials):
    """
    Kiểm tra xem file có tên filename đã tồn tại trong folder_id trên Drive hay chưa.
    Trả về True nếu tồn tại, False nếu chưa.
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


# --- HÀM: TÌM HOẶC TẠO FOLDER CON TRÊN DRIVE ---
@st.cache_resource(show_spinner="Đang kiểm tra/tạo folder Drive...")
def get_or_create_drive_folder(parent_id, folder_name, _credentials):
    """
    Tìm ID của folder con trong parent_id. Nếu chưa tồn tại, tạo mới.
    Trả về ID của folder con.
    """
    try:
        service = build('drive', 'v3', credentials=_credentials)
        
        # 1. Tìm kiếm folder
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
            # 2. Tạo folder mới nếu chưa tồn tại
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
        
# --- HÀM HỖ TRỢ HIỂN THỊ ẢNH DATASET (ĐÃ THÊM) ---
def load_dataset_image(stt_match, dataset_folder):
    """
    Tìm và trả về đường dẫn của ảnh dataset tương ứng với STT match đầu tiên.
    """
    pattern_simple = re.compile(rf'^{stt_match}\.jpe?g$', re.IGNORECASE)
    pattern_complex = re.compile(rf'^{stt_match}_.*\.jpe?g$', re.IGNORECASE)
    
    if os.path.isdir(dataset_folder):
        for filename in os.listdir(dataset_folder):
            
            if pattern_simple.match(filename) or pattern_complex.match(filename):
                return os.path.join(dataset_folder, filename)
                
    return None
        
# --- LOGIC GHI DỮ LIỆU VÀ LƯU ẢNH MỚI ---
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


# --- HÀM: CẬP NHẬT PLACEHOLDER CHECKLIST ---
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
    
    # === KHỞI TẠO KEY SESSION STATE ===
    if 'camera_input_key' not in st.session_state:
        st.session_state['camera_input_key'] = 0
    # =================================

    # 1. Tải Dataset & Checklist
    from config import GDRIVE_DATASET_FOLDER_ID
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

    # --- BỔ SUNG: CHECKBOX HIỂN THỊ ẢNH DEBUG ---
    show_debug_images = st.checkbox(
        "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
        value=True, 
        help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset (khi điểm danh thành công) hoặc ảnh đã cắt (khi không khớp)."
    )
    # ---------------------------------------------

    st.markdown("---")

    # 3. Chụp Ảnh và Xử Lý
    if selected_session:
        
        # TẠO CỘT MỚI: [Tỉ lệ 1, Tỉ lệ 2] => Camera chiếm 1/3 chiều rộng
        col_camera, col_spacer = st.columns([1, 1])
        
        with col_camera:
            # ĐẶT CAMERA INPUT VÀO CỘT CÓ KÍCH THƯỚC GIỚI HẠN
            captured_file = st.camera_input(
                "Chụp ảnh điểm danh", 
                key=f"camera_input_{st.session_state['camera_input_key']}" 
            )
        
        # Tạo placeholder cho kết quả (Đảm bảo nó vẫn nằm ngoài col_camera)
        result_placeholder = st.empty()
        
        TEMP_IMAGE_PATH = None

        if captured_file is not None:
            
            image_bytes_original = captured_file.getvalue() 
            
            stt_match = None
            distance = None
            

            with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
                
                # --- SỬ DỤNG HÀM PHÁT HIỆN MẠNH MẼ MỚI (DeepFace) ---
                processed_image_np, face_detected, num_faces, TEMP_IMAGE_PATH = robust_detect_and_crop_face(image_bytes_original)
                processed_image = Image.fromarray(processed_image_np)
                
                # Kiểm tra chỉ có 1 khuôn mặt và tiến hành so khớp
                if face_detected and num_faces == 1 and TEMP_IMAGE_PATH:
                    
                    stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)
                
                # --- End If face_detected and num_faces == 1 ---
                
            # HIỂN THỊ KẾT QUẢ TRONG PLACEHOLDER
            with result_placeholder.container():
                st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
                st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu (Xanh: 1 khuôn mặt; Đỏ: >1 khuôn mặt).", width='stretch')

                st.markdown("---")
                st.subheader("💡 Kết quả Điểm danh")
                
                if stt_match and distance is not None: 
                    st.balloons()
                    st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
                    
                    # --- HIỂN THỊ ẢNH DEBUG (CÓ ĐIỀU KIỆN) ---
                    if show_debug_images: 
                        dataset_image_path = load_dataset_image(stt_match, DATASET_FOLDER)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                                st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", width='stretch')
                            
                        with col2:
                            if dataset_image_path:
                                st.image(dataset_image_path, caption=f"Dataset (STT: {stt_match})", width='stretch')
                            else:
                                st.warning("Không tìm thấy ảnh dataset để hiển thị.")
                    # ----------------------------------------------------------------------------
                    
                    st.markdown(f"""
                    * **STT trùng khớp:** **{stt_match}**
                    * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
                    """)
                    
                    # Cập nhật checklist VÀ LƯU ẢNH GỐC THÀNH CÔNG
                    updated = update_checklist_and_save_new_data(stt_match, selected_session, image_bytes_original, credentials)
                    
                    if updated and CHECKLIST_SESSION_KEY in st.session_state:
                         update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
                    
                    if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                        os.remove(TEMP_IMAGE_PATH)
                        
                    st.session_state['camera_input_key'] += 1 
                    st.rerun() 
                    return 
                    
                elif face_detected and num_faces == 1:
                    st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
                    
                    if show_debug_images: 
                        if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                            st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", width='content')
                    
                    update_checklist_and_save_new_data(None, selected_session, image_bytes_original, credentials) 
                    
                elif face_detected and num_faces > 1:
                    st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

                else:
                    st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
                    st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

            if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
                
    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU HOẶC SAU KHI RERUN
    if CHECKLIST_SESSION_KEY in st.session_state:
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
