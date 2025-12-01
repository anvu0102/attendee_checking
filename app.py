import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
from deepface import DeepFace

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Sử dụng camera để chụp ảnh, nhận diện và đối chiếu với dataset bằng DeepFace.")

# --- 2. Tải và Thiết lập Haar Cascade (Dùng cho phát hiện khung, không dùng cho so khớp) ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
face_cascade = None
TEMP_IMAGE_PATH = "captured_face.jpg" # Đường dẫn tạm để lưu ảnh chụp
DATASET_FOLDER = "dataset" # Thư mục chứa các khuôn mặt đã đăng ký

@st.cache_resource
def load_face_cascade(url, filename):
    """ Tải Haar Cascade từ URL và lưu trữ trong bộ nhớ đệm của Streamlit. """
    try:
        # Tải từ GitHub (giống như code gốc)
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            classifier = cv2.CascadeClassifier(filename)
            if not classifier.empty():
                st.success("Tải Haar Cascade thành công.")
                return classifier
            
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

# Khởi tạo bộ phân loại
face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)

# Đảm bảo thư mục dataset tồn tại
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)
    st.warning(f"Đã tạo thư mục '{DATASET_FOLDER}'. Vui lòng thêm ảnh khuôn mặt đã đăng ký vào đây.")


# --- 3. Hàm Phát hiện Khuôn mặt (Giữ nguyên để vẽ khung) ---
def detect_and_draw_face(image_bytes, cascade):
    """
    Nhận diện khuôn mặt trên ảnh đầu vào, vẽ khung, và trả về ảnh đã xử lý 
    cùng với cờ (flag) cho biết có khuôn mặt hay không.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

    # Vẽ khung vuông lên ảnh
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, faces, image_bgr # Thêm image_bgr để lưu file


# --- 4. Hàm DeepFace Recognition (Hàm mới) ---
@st.cache_data
def verify_face_against_dataset(target_image_path, dataset_folder):
    """
    Sử dụng DeepFace để so sánh ảnh đầu vào với tất cả ảnh trong dataset.
    Trả về tên người khớp (tên file) hoặc None.
    """
    try:
        # Chạy DeepFace.find để tìm tất cả các khuôn mặt khớp
        # model_name='ArcFace' và distance_metric='cosine' là các tham số phổ biến
        # distance_threshold có thể cần chỉnh sửa (ArcFace cosine: 0.68)
        
        # NOTE: DeepFace.find() trả về một list các DataFrames. Ta chỉ quan tâm kết quả đầu tiên.
        df = DeepFace.find(
            img_path=target_image_path, 
            db_path=dataset_folder, 
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True # Yêu cầu phát hiện khuôn mặt để so khớp
        )
        
        # Nếu DataFrame không rỗng (tìm thấy kết quả khớp)
        if isinstance(df, list) and len(df) > 0 and not df[0].empty:
            # Lấy dòng đầu tiên (khớp tốt nhất - khoảng cách nhỏ nhất)
            best_match = df[0].iloc[0]
            # Lấy tên file gốc từ cột 'identity'
            identity_path = best_match['identity']
            # Tên người là tên file (trước dấu chấm)
            person_name = os.path.splitext(os.path.basename(identity_path))[0]
            distance = best_match['ArcFace_cosine'] # Khoảng cách so khớp
            return person_name, distance
        
        return None, None
    
    except ValueError as e:
        # DeepFace ném ValueError nếu không tìm thấy khuôn mặt trong ảnh đầu vào
        if "Face could not be detected" in str(e):
             st.error("❌ Không phát hiện khuôn mặt trong ảnh chụp. Vui lòng thử lại.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình so khớp DeepFace: {e}")
        return None, None


# --- 5. Giao diện và Luồng Ứng dụng ---
st.info(f"Nhấn 'Chụp ảnh' để Streamlit truy cập camera. **Yêu cầu:** Thư mục '{DATASET_FOLDER}' phải chứa ảnh khuôn mặt đã đăng ký (ví dụ: 'NguyenVanA.jpg').")

# Sử dụng widget camera_input của Streamlit
captured_file = st.camera_input("Chụp ảnh điểm danh:")

if captured_file is not None:
    if face_cascade is None:
        st.error("Không thể tiếp tục do lỗi tải bộ phân loại khuôn mặt. Vui lòng kiểm tra nhật ký.")
    else:
        # Đọc bytes của ảnh
        image_bytes = captured_file.getvalue()
        
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            # 1. Phát hiện khuôn mặt và vẽ khung
            processed_image_np, face_detected, face_locations, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
            
            # Chuyển mảng NumPy về đối tượng Image để hiển thị
            processed_image = Image.fromarray(processed_image_np)
            
            # 2. Lưu ảnh tạm thời để DeepFace sử dụng (DeepFace cần đường dẫn file)
            # Lưu ảnh BGR OpenCV vào đường dẫn tạm
            cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
            
            # 3. Thực hiện so khớp DeepFace
            match_name, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

        # Xóa file tạm sau khi đã xử lý
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu bằng khung màu xanh dương.", use_column_width=True)

        # Kiểm tra kết quả
        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if match_name:
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **Người trùng khớp:** **{match_name}**
            * **Khoảng cách Cosine (DeepFace ArcFace):** {distance:.4f}
            * *Giả định:* Khoảng cách nhỏ hơn ngưỡng (mặc định ~0.68) => Khớp.
            """)
            
        elif face_detected and match_name is None:
            st.warning(f"⚠️ **Phát hiện {len(face_locations)} khuôn mặt, nhưng không khớp với dataset.**")
            st.markdown("""
            * Vui lòng kiểm tra lại ảnh trong thư mục `dataset`.
            * Thử chụp lại ảnh với điều kiện ánh sáng tốt hơn.
            """)
            
        else: # face_detected is False (và match_name is None)
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình, với đủ ánh sáng.")
