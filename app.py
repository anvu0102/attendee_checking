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
import pandas as pd # Thêm thư viện pandas để xử lý file checklist

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace (GDrive)",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Dataset và Checklist được tải từ Google Drive công khai.")

# --- 2. Cấu hình & Hằng số ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# VUI LÒNG THAY THẾ CÁC ID DƯỚI ĐÂY BẰNG ID THỰC TẾ CỦA BẠN
GDRIVE_DATASET_ID = "1-yAtAUD5FY69hlLYP_O3pfqRzKgompcd" # ID cho file ZIP dataset
ZIP_FILENAME = "dataset_archive.zip" 
DATASET_FOLDER = "dataset" 

GDRIVE_CHECKLIST_ID = "1lcVBJZ55nQVoQYi6PK0iUV5Y_cCY74lv" # ID cho file CSV checklist
CHECKLIST_FILENAME = "checklist.csv" 
CHECKLIST_SESSION_KEY = "attendance_df" 

DETECTOR_BACKEND = "opencv"
NEW_DATA_FOLDER = "new_data" # Thư mục local để lưu ảnh mới


# --- Hàm tải file từ Google Drive ---
def download_file_from_gdrive(file_id, output_filename):
    """ Tải file công khai từ Google Drive. """
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status() 
        
        if "confirm" in response.headers.get("Content-Disposition", ""):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = requests.get(DOWNLOAD_URL, params=params, stream=True)
                    response.raise_for_status()
                    break

        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi tải file {output_filename} từ Drive: {e}")
        return False


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


@st.cache_resource(show_spinner="Đang tải và giải nén Dataset từ Google Drive (Chỉ chạy lần đầu)...")
def download_and_extract_dataset(file_id, zip_name, target_folder):
    """ Tải và giải nén dataset ZIP. """
    if file_id == "YOUR_GDRIVE_FILE_ID_HERE":
        return False
        
    deepface_cache = os.path.join(target_folder, 'representations_arcface.pkl')
    if os.path.exists(deepface_cache) and os.path.isdir(target_folder) and len(os.listdir(target_folder)) > 1:
         st.success(f"Dataset đã sẵn sàng tại '{target_folder}'. Bỏ qua tải xuống.")
         return True
    
    st.info(f"Đang tải dataset từ Google Drive File ID: {file_id}...")
    
    if download_file_from_gdrive(file_id, zip_name):
        try:
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                zip_ref.extractall(".")
            st.success(f"Giải nén thành công vào thư mục '{target_folder}'.")
            if os.path.exists(zip_name):
                os.remove(zip_name)
            return True
        except Exception as e:
            st.error(f"❌ Lỗi khi giải nén: {e}")
            return False
    return False

@st.cache_data(show_spinner="Đang tải và xử lý Checklist từ Google Drive...")
def load_checklist(file_id, filename):
    """ Tải checklist CSV/Excel và đọc thành DataFrame. """
    if file_id == "YOUR_GDRIVE_CHECKLIST_ID_HERE":
        return None
    
    # Kiểm tra xem file đã được tải/tạo chưa, nếu không thì tải từ Drive
    if not os.path.exists(filename):
        download_file_from_gdrive(file_id, filename)
        
    if os.path.exists(filename):
        try:
            # Giả định file checklist là CSV
            df = pd.read_csv(filename)
            # Không xóa file để giữ lại phiên bản gốc nếu có lỗi cập nhật
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng CSV.")
            return None
    return None

# --- 3. Hàm Phát hiện Khuôn mặt (Dùng cho hiển thị khung) ---
def detect_and_draw_face(image_bytes, cascade):
    """
    Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung vuông lên ảnh
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, len(faces), image_bgr


# --- 4. Hàm DeepFace Recognition ---
def verify_face_against_dataset(target_image_path, dataset_folder):
    """
    Sử dụng DeepFace để so sánh ảnh đầu vào với dataset.
    Trả về STT khớp (tên file) và khoảng cách.
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
            # Lấy STT (tên file)
            stt_match = os.path.splitext(os.path.basename(identity_path))[0] 
            distance = best_match['ArcFace_cosine'] 
            return stt_match, distance
        
        return None, None
    
    except ValueError as e:
        if "Face could not be detected" in str(e):
             # DeepFace.find() sẽ ném ValueError nếu không tìm thấy khuôn mặt
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. Vui lòng thử lại ảnh rõ ràng hơn.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình so khớp DeepFace: {e}")
        return None, None

# --- 5. Logic Ghi Dữ Liệu (Mô phỏng ghi lên Drive) ---

def update_checklist_and_save_new_data(stt_match, captured_image_bgr, session_name, image_bytes):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới (hoặc mô phỏng).
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return

    df = st.session_state[CHECKLIST_SESSION_KEY]
    
    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        # stt_match là tên file (ví dụ: '111'), tương ứng với cột STT/MSSV trong checklist
        try:
            # Lấy tên cột đầu tiên (ví dụ: 'Stt')
            stt_col = df.columns[0] 
            
            # Giả định STT/MSSV trong file name trùng với STT trong file checklist (cột đầu tiên)
            # Chuyển sang string để tìm kiếm chính xác
            stt_match_str = str(stt_match).split('_')[0] # Lấy phần STT trước dấu gạch dưới nếu có (VD: 111_ten -> 111)
            
            # Tìm chỉ số dòng của STT. Sử dụng .str.contains để linh hoạt hơn
            row_index = df[df[stt_col].astype(str).str.contains(stt_match_str, regex=False)].index
            
            if not row_index.empty:
                # Cập nhật cột Buổi được chọn
                df.loc[row_index[0], session_name] = 'X'
                st.session_state[CHECKLIST_SESSION_KEY] = df # Cập nhật Session State
                
                # --- Mô phỏng ghi lên Drive ---
                st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")
                st.info("⚠️ **Mô phỏng:** Trong ứng dụng thực tế, DataFrame này cần được ghi trở lại file Drive (ví dụ: ghi lại file CSV/Excel lên Drive).")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match_str}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới (Nếu không khớp)
    else: 
        # Nếu không khớp (stt_match is None), tiến hành lưu ảnh mới
        
        # Tìm số thứ tự tiếp theo cho ảnh mới
        if 'new_data_counter' not in st.session_state:
            st.session_state['new_data_counter'] = 0
            
        st.session_state['new_data_counter'] += 1
        new_counter = st.session_state['new_data_counter']
        
        # Tên file: B<Số Buổi>_<Counter> (VD: B1_1.jpg)
        session_num = session_name.replace("Buổi ", "")
        new_filename = f"B{session_num}_{new_counter}.jpg" 
        
        # Tạo thư mục nếu chưa có
        if not os.path.exists(NEW_DATA_FOLDER):
            os.makedirs(NEW_DATA_FOLDER)
            
        new_filepath = os.path.join(NEW_DATA_FOLDER, new_filename)
        
        # Lưu ảnh gốc dưới dạng JPG
        image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_to_save.save(new_filepath, format='JPEG')
        
        # --- Mô phỏng ghi lên Drive ---
        st.success(f"✅ **Đã lưu ảnh mới** vào: **{NEW_DATA_FOLDER}/{new_filename}**")
        st.info("⚠️ **Mô phỏng:** Trong ứng dụng thực tế, ảnh này cần được tải lên thư mục Drive.")


# --- 6. Giao diện và Luồng Ứng dụng ---

# 6.1 KHỞI TẠO VÀ TẢI DATASET & CHECKLIST
dataset_ready = download_and_extract_dataset(GDRIVE_DATASET_ID, ZIP_FILENAME, DATASET_FOLDER)
checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME)

# Lưu checklist vào session state để có thể cập nhật
if checklist_df is not None:
    st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
    
st.markdown("---")

if not dataset_ready:
     st.warning("⚠️ Vui lòng cấu hình đúng File ID ZIP Dataset và thử lại.")
     st.stop()
     
if checklist_df is None:
     st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và định dạng CSV.")
     st.stop()


st.info(f"Dataset đã tải xong. Checklist có {len(checklist_df)} người.")


# 6.2 CHỌN BUỔI HỌC (Dropdown)
attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

if not attendance_cols:
     st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file.")
     st.stop()

selected_session = st.selectbox(
    "1️⃣ **Chọn Buổi Điểm Danh**", 
    attendance_cols, 
    index=0,
    help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
)
st.success(f"Đang điểm danh cho: **{selected_session}**")

st.markdown("---")

# 6.3 CHỤP ẢNH VÀ XỬ LÝ
captured_file = st.camera_input("2️⃣ Chụp ảnh điểm danh:")

if captured_file is not None:
    if face_cascade is None:
        st.error("Không thể tiếp tục do lỗi tải bộ phân loại khuôn mặt.")
    else:
        image_bytes = captured_file.getvalue()
        
        # Mở spinner trong lúc xử lý
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            
            # 1. Phát hiện khuôn mặt và vẽ khung
            processed_image_np, face_detected, num_faces, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
            processed_image = Image.fromarray(processed_image_np)
            
            # 2. LƯU ẢNH TẠM THỜI DUY NHẤT (QUAN TRỌNG: Dùng tempfile)
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            TEMP_IMAGE_PATH = temp_file.name
            temp_file.close() 
            
            # Ghi ảnh BGR vào file tạm để DeepFace xử lý
            cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
            
            # 3. Thực hiện so khớp DeepFace
            stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

        # Xóa file tạm sau khi đã xử lý xong
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if stt_match:
            # 4a. Khuôn mặt khớp -> Cập nhật checklist
            st.balloons()
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **STT trùng khớp:** **{stt_match}**
            * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
            """)
            # Truyền None cho captured_image_bgr vì đây là trường hợp khớp
            update_checklist_and_save_new_data(stt_match, None, selected_session, None)
            
        elif face_detected and num_faces == 1:
            # 4b. 1 khuôn mặt KHÔNG khớp -> Lưu ảnh mới
            st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
            # Truyền image_bytes để lưu ảnh gốc
            update_checklist_and_save_new_data(None, image_bgr, selected_session, image_bytes) 
            
        elif face_detected and num_faces > 1:
            # 4c. Nhiều khuôn mặt
            st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

        else:
            # 4d. Không phát hiện
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

st.markdown("---")
st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
if CHECKLIST_SESSION_KEY in st.session_state:
    st.dataframe(st.session_state[CHECKLIST_SESSION_KEY])
