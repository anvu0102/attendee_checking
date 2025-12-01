# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
Đã bổ sung: Checkbox để điều khiển việc hiển thị ảnh đã cắt và ảnh dataset/không khớp.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io # Import io cho việc xử lý file trong bộ nhớ
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
            # st.success("✅ Haar Cascade đã sẵn sàng.")
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
    Trả về: ảnh có khung (RGB), ảnh gốc (BGR), cờ phát hiện, số lượng khuôn mặt, TỌA ĐỘ (x,y,w,h).
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
    Lưu ý: Vì ảnh đã được cắt và lưu, ta đặt enforce_detection=False để DeepFace không cần tìm lại.
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
            # KHÔNG CẦN CẮT NỮA VÌ ẢNH ĐÃ ĐƯỢC CẮT BÊN NGOÀI
        )
        
        # Kiểm tra nếu có kết quả và DataFrame đầu tiên không rỗng
        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            best_match = df_list[0].iloc[0]
            identity_path = best_match['identity']
            print(identity_path)
            # Lấy STT từ tên file (vd: 1_001.jpg -> 1)
            stt_match = os.path.splitext(os.path.basename(identity_path))[0].split('_')[0]
            distance = best_match['ArcFace_cosine'] 
            
            # Đảm bảo distance là float trước khi trả về
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
    # Chúng ta chỉ quan tâm đến phần số cuối cùng trước .jpg
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
                # Bỏ qua nếu không phải là số
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
            # Đã tìm thấy
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
        
# --- HÀM GHI ĐÈ FILE CHECKLIST LÊN DRIVE BẰNG ID (KHÔNG DÙNG) ---
def overwrite_gdrive_checklist_file(local_path, file_id, _credentials):
    # Hàm này không được sử dụng
    pass

# --- HÀM HỖ TRỢ HIỂN THỊ ẢNH DATASET (ĐÃ THÊM) ---
def load_dataset_image(stt_match, dataset_folder):
    """
    Tìm và trả về đường dẫn của ảnh dataset tương ứng với STT match đầu tiên.
    Đã cập nhật regex để hỗ trợ cả định dạng STT.jpg và STT_*.jpg.
    """
    # Sử dụng hai pattern riêng biệt để linh hoạt hơn:
    pattern_simple = re.compile(rf'^{stt_match}\.jpe?g$', re.IGNORECASE)
    pattern_complex = re.compile(rf'^{stt_match}_.*\.jpe?g$', re.IGNORECASE)
    
    if os.path.isdir(dataset_folder):
        for filename in os.listdir(dataset_folder):
            
            # 1. Kiểm tra định dạng đơn giản (c.jpg)
            if pattern_simple.match(filename):
                return os.path.join(dataset_folder, filename)
                
            # 2. Kiểm tra định dạng phức tạp (c_001.jpg)
            if pattern_complex.match(filename):
                return os.path.join(dataset_folder, filename)
                
    return None
        
# --- LOGIC GHI DỮ LIỆU VÀ LƯU ẢNH MỚI (ĐÃ CẬP NHẬT) ---
def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, _credentials):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới lên Drive.
    
    Lưu ý: image_bytes ở đây luôn là bytes của ảnh GỐC từ camera.
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return False # Trả về False nếu lỗi

    df = st.session_state[CHECKLIST_SESSION_KEY]
    updated = False # Biến cờ cho biết DF có được cập nhật không
    
    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        try:
            stt_col = df.columns[0] 
            
            # Tìm dòng khớp STT
            row_index = df[df[stt_col].astype(str).str.contains(stt_match, regex=False)].index
            
            if not row_index.empty:
                
                # --- LƯU ẢNH GỐC VÀO FOLDER THEO BUỔI (Điểm danh thành công) ---
                stt = df.loc[row_index[0], stt_col]
                session_folder_name = session_name.replace("Buổi ", "B")
                
                # 1. Tìm hoặc tạo folder con trong GDRIVE_NEW_DATA_FOLDER_ID
                target_folder_id = get_or_create_drive_folder(
                    GDRIVE_NEW_DATA_FOLDER_ID, 
                    session_folder_name, 
                    _credentials
                )
                
                if target_folder_id:
                    # 2. Xây dựng tên file gốc và kiểm tra tồn tại
                    base_filename = f"{session_folder_name}_{stt}.jpg" 
                    drive_filename = base_filename # Tên file mặc định

                    if check_drive_file_existence(target_folder_id, base_filename, _credentials):
                        # Nếu file đã tồn tại, thêm timestamp để phân biệt
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
                        
                        # Upload ảnh vào folder con
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
                    updated = True # Đánh dấu đã cập nhật
                    
                    st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")

                else:
                    st.info(f"Người có STT **{df.loc[row_index[0], stt_col]}** đã được điểm danh trong **{session_name}**.")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp) - SỬ DỤNG ẢNH GỐC
    else: 
        # Cảnh báo không khớp
        st.warning("⚠️ Đang lưu ảnh vào folder dữ liệu mới...")
        
        # --- LOGIC LƯU ẢNH GỐC KHÔNG KHỚP (GIỮ NGUYÊN) ---
        # Lấy số thứ tự tiếp theo dựa trên các file hiện có trên Drive
        next_counter = get_next_new_data_stt(_credentials)
        
        # Tạo tên file theo định dạng B<buổi>_<counter>.jpg
        session_num = session_name.replace("Buổi ", "")
        drive_filename = f"B{session_num}_{next_counter}.jpg" 
        
        # --- TẠO FILE TẠM ĐỂ UPLOAD ---
        temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_UPLOAD_PATH = temp_file_for_upload.name
        temp_file_for_upload.close()
        
        try:
            # image_bytes ở đây là ảnh gốc (full image)
            image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
            
            # Gọi hàm Upload Drive (REAL) - Truyền _credentials
            upload_to_gdrive_real(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, _credentials)
            st.info(f"🖼️ Đã lưu ảnh không khớp vào folder chung: {drive_filename}")

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)
        # ----------------------------------------------------------
                
    return updated # Trả về cờ cập nhật


# --- HÀM: CẬP NHẬT PLACEHOLDER CHECKLIST ---
def update_checklist_display(checklist_placeholder, current_df):
    """Cập nhật nội dung của placeholder checklist."""
    with checklist_placeholder.container():
        st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
        st.dataframe(current_df)
        
        # Tạo file Excel trong bộ nhớ (sử dụng io.BytesIO)
        output = io.BytesIO()
        current_df.to_excel(output, index=False, sheet_name='Checklist_Cap_Nhat')
        excel_data = output.getvalue()
        
        # Hiển thị nút tải về
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
            # st.info("✅ Đã tải Checklist từ Drive vào Session State.")
        else:
            # Xử lý lỗi tải lần đầu
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

    st.info("**Vui lòng chọn một Buổi Điểm Danh để tiếp tục.**")
    
    # --- THAY ĐỔI: Thêm một tùy chọn mặc định không phải là buổi học ---
    display_options = ["--- Vui lòng chọn buổi ---"] + attendance_cols
    
    selected_session_display = st.selectbox(
        "", 
        display_options, 
        index=0, # Mặc định chọn tùy chọn đầu tiên ("--- Vui lòng chọn buổi ---")
    )
    
    # Xác định buổi học thực sự được chọn
    selected_session = selected_session_display if selected_session_display != "--- Vui lòng chọn buổi ---" else None

    # --- BỔ SUNG: CHECKBOX HIỂN THỊ ẢNH DEBUG ---
    show_debug_images = st.checkbox(
        "Hiển thị Ảnh đã Cắt và Ảnh Dataset",
        value=True, # Mặc định bật
        help="Bật để xem ảnh khuôn mặt được cắt ra và ảnh tương ứng trong dataset (khi điểm danh thành công) hoặc ảnh đã cắt (khi không khớp)."
    )
    # ---------------------------------------------

    st.markdown("---")

    # 3. Chụp Ảnh và Xử Lý
    # --- THAY ĐỔI: Chỉ hiển thị camera input nếu đã chọn buổi ---
    if selected_session:
        
        # --- THÊM KEY VÀO CAMERA INPUT ---
        captured_file = st.camera_input(
            "Chụp ảnh điểm danh", 
            key=f"camera_input_{st.session_state['camera_input_key']}" # Sử dụng key từ session state
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
                    if show_debug_images: # <<< KIỂM TRA CHECKBOX
                        dataset_image_path = load_dataset_image(stt_match, DATASET_FOLDER)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Hiển thị ảnh đã cắt (đã lưu tạm thời)
                            # TEMP_IMAGE_PATH chỉ tồn tại nếu phát hiện 1 khuôn mặt
                            if TEMP_IMAGE_PATH:
                                st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=True)
                            
                        with col2:
                            if dataset_image_path:
                                # Hiển thị ảnh dataset trùng khớp
                                st.image(dataset_image_path, caption=f"Dataset (STT: {stt_match})", use_column_width=True)
                            else:
                                st.warning("Không tìm thấy ảnh dataset để hiển thị.")
                    # ----------------------------------------------------------------------------
                    
                    st.markdown(f"""
                    * **STT trùng khớp:** **{stt_match}**
                    * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
                    """)
                    
                    # Cập nhật checklist VÀ LƯU ẢNH GỐC THÀNH CÔNG
                    # TRUYỀN BYTES CỦA ẢNH GỐC
                    updated = update_checklist_and_save_new_data(stt_match, selected_session, image_bytes_original, credentials)
                    
                    # --- HIỂN THỊ CHECKLIST ĐÃ CẬP NHẬT TRƯỚC KHI RERUN ---
                    if updated and CHECKLIST_SESSION_KEY in st.session_state:
                         # Nếu có cập nhật, vẽ lại bảng ngay lập tức
                         update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
                    # ----------------------------------------------------
                    
                    # --- LOGIC TỰ ĐỘNG CLEAR ---
                    time.sleep(2) # Đợi 2 giây
                    
                    # Xóa file tạm sau khi đã hiển thị xong (trước khi rerun)
                    if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                        os.remove(TEMP_IMAGE_PATH)
                        
                    # Tăng giá trị key để buộc Streamlit reset widget st.camera_input
                    st.session_state['camera_input_key'] += 1 
                    st.rerun() # Buộc rerun
                    # --------------------------------------
                    return 
                    
                elif face_detected and num_faces == 1:
                    st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
                    
                    # --- BỔ SUNG HIỂN THỊ ẢNH ĐÃ CẮT (CÓ ĐIỀU KIỆN) ---
                    if show_debug_images: # <<< KIỂM TRA CHECKBOX
                        # Ảnh đã cắt được tạo và lưu ở TEMP_IMAGE_PATH
                        if TEMP_IMAGE_PATH:
                            st.image(TEMP_IMAGE_PATH, caption="Khuôn mặt đã Cắt (Cropped)", use_column_width=False)
                    # ----------------------------------------------------
                    
                    # Lưu ảnh gốc (truyền image_bytes_original)
                    update_checklist_and_save_new_data(None, selected_session, image_bytes_original, credentials) 
                    
                elif face_detected and num_faces > 1:
                    st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

                else:
                    st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
                    st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

            # --- Vị trí XÓA file tạm mới: Xóa file tạm nếu không vào khối logic tự động clear 5s ---
            if TEMP_IMAGE_PATH and os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
            # ---------------------------------------------------------------------------------------
                
            # --- End result_placeholder.container() ---
            
    # 4. HIỂN THỊ TRẠNG THÁI CHECKLIST BAN ĐẦU HOẶC SAU KHI RERUN
    if CHECKLIST_SESSION_KEY in st.session_state:
        update_checklist_display(checklist_placeholder, st.session_state[CHECKLIST_SESSION_KEY])
