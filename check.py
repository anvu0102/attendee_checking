# check.py
"""
Chứa các hàm xử lý DeepFace, OpenCV, logic cập nhật checklist và giao diện Streamlit.
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
            st.success("✅ Haar Cascade đã sẵn sàng.")
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
    Trả về: ảnh có khung (RGB), ảnh gốc (BGR), cờ phát hiện, số lượng khuôn mặt.
    """
    
    # Đọc ảnh từ bytes
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    # Lấy ảnh gốc BGR để truyền cho DeepFace
    image_original_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    
    # Tạo bản sao để vẽ khung
    image_bgr_with_frame = image_original_bgr.copy()
    
    gray = cv2.cvtColor(image_original_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        # Phát hiện khuôn mặt
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung lên bản sao
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr_with_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr_with_frame, cv2.COLOR_BGR2RGB)

    # TRẢ VỀ: (ảnh có khung (RGB), ảnh GỐC (BGR), cờ phát hiện, số lượng khuôn mặt)
    return processed_image_rgb, image_original_bgr, len(faces) > 0, len(faces)


def verify_face_against_dataset(target_image_path, dataset_folder):
    """ Sử dụng DeepFace để so sánh ảnh đầu vào với dataset. """
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
            return stt_match, distance
        return None, None
    except Exception as e:
        # Chỉ in lỗi DeepFace nếu không phải lỗi không phát hiện
        if "Face could not be detected" in str(e):
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. (Kiểm tra chất lượng ảnh)")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None


@st.cache_data(show_spinner="Đang tải và xử lý Checklist (XLSX) từ Google Drive...")
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

# --- LOGIC GHI DỮ LIỆU VÀ LƯU ẢNH MỚI (ĐÃ CẬP NHẬT) ---
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
            
            # Tìm dòng khớp STT
            row_index = df[df[stt_col].astype(str).str.contains(stt_match, regex=False)].index
            
            if not row_index.empty:
                # Kiểm tra nếu chưa điểm danh thì mới cập nhật
                if df.loc[row_index[0], session_name] != 'X':
                    df.loc[row_index[0], session_name] = 'X'
                    st.session_state[CHECKLIST_SESSION_KEY] = df 
                    
                    st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")
                    st.info(f"⚠️ **Cần thêm chức năng ghi ngược (Write-Back) DataFrame này lên file XLSX Drive ID: {GDRIVE_CHECKLIST_ID}**.")
                else:
                    st.info(f"Người có STT **{df.loc[row_index[0], stt_col]}** đã được điểm danh trong **{session_name}**.")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp)
    else: 
        # Cảnh báo không khớp
        st.warning("⚠️ Khuôn mặt không khớp. Đang lưu ảnh vào folder dữ liệu mới...")

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
            image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
            
            # Gọi hàm Upload Drive (REAL) - Truyền _credentials
            upload_to_gdrive_real(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, _credentials)

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)


# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    
    # 1. Tải Dataset & Checklist
    from config import GDRIVE_DATASET_FOLDER_ID, GDRIVE_CHECKLIST_ID
    from config import download_dataset_folder_real
    
    # Tải Folder Dataset (REAL) - Truyền CREDENTIALS vào tham số _credentials
    dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials) 
    # Tải Checklist (XLSX) - Truyền CREDENTIALS vào tham số _credentials
    checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, credentials)

    if checklist_df is not None:
        st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
        
    st.markdown("---")

    if not dataset_ready:
         st.warning("⚠️ Lỗi tải Dataset Folder. Vui lòng kiểm tra ID Drive Folder và quyền truy cập.")
         return
         
    if checklist_df is None:
         st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
         return

    st.info(f"Dataset đã sẵn sàng. Checklist có {len(checklist_df)} người.")

    # 2. Chọn Buổi Học (Dropdown)
    attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

    if not attendance_cols:
         st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
         return

    selected_session = st.selectbox(
        "1️⃣ **Chọn Buổi Điểm Danh**", 
        attendance_cols, 
        index=0,
        help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
    )
    st.success(f"Đang điểm danh cho: **{selected_session}**")

    st.markdown("---")

    # 3. Chụp Ảnh và Xử Lý
    captured_file = st.camera_input("2️⃣ Chụp ảnh điểm danh:")

    if captured_file is not None:
        
        image_bytes = captured_file.getvalue()
        
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            
            # Phát hiện khuôn mặt và vẽ khung
            # NHẬN KẾT QUẢ GỒM: ảnh có khung (RGB), ảnh GỐC (BGR), cờ phát hiện, số lượng khuôn mặt
            processed_image_np, image_original_bgr, face_detected, num_faces = detect_and_draw_face(image_bytes, face_cascade)
            processed_image = Image.fromarray(processed_image_np)
            
            # LƯU ẢNH GỐC (chưa vẽ khung) TẠM THỜI cho DeepFace so khớp
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            TEMP_IMAGE_PATH = temp_file.name
            temp_file.close() 
            
            # LƯU ẢNH GỐC BGR
            cv2.imwrite(TEMP_IMAGE_PATH, image_original_bgr)
            
            # Thực hiện so khớp DeepFace
            stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

        # Xóa file tạm
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")

        stt_match = 2
        if stt_match:
            st.balloons()
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **STT trùng khớp:** **{stt_match}**
            * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
            """)
            # Cập nhật checklist (truyền credentials)
            update_checklist_and_save_new_data(stt_match, selected_session, None, credentials)
            
        elif face_detected and num_faces == 1:
            st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
            # Lưu ảnh mới (truyền image_bytes và credentials)
            update_checklist_and_save_new_data(None, selected_session, image_bytes, credentials) 
            
        elif face_detected and num_faces > 1:
            st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

        else:
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

    st.markdown("---")
    st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
    if CHECKLIST_SESSION_KEY in st.session_state:
        st.dataframe(st.session_state[CHECKLIST_SESSION_KEY])
