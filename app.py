import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt (Streamlit)")
st.caption("Sử dụng camera để chụp ảnh và mô phỏng quá trình nhận dạng.")

# --- 2. Hàm Nhận diện Khuôn mặt (Sử dụng OpenCV) ---

# Tải bộ phân loại khuôn mặt Haar Cascade (file này cần có trong thư mục)
# Để đơn giản, bạn có thể tải file này và đặt cùng thư mục với app.py
# Link: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    # Nếu không tìm thấy file, sẽ báo lỗi nhưng vẫn cho phép chụp ảnh
    st.error("Không tìm thấy tệp haarcascade_frontalface_default.xml. Nhận diện khuôn mặt sẽ không hoạt động.")
    face_cascade = None


def detect_and_draw_face(image_bytes):
    """
    Nhận diện khuôn mặt trên ảnh đầu vào, vẽ khung, và trả về ảnh đã xử lý 
    cùng với cờ (flag) cho biết có khuôn mặt hay không.
    """
    # Chuyển đổi bytes thành mảng NumPy
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    
    # Chuyển sang ảnh xám để nhận diện
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    faces = []
    if face_cascade is not None:
        # Thực hiện nhận diện khuôn mặt
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

    # Vẽ khung vuông lên ảnh gốc
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return image_np, len(faces) > 0, faces

# --- 3. Giao diện và Luồng Ứng dụng ---

st.info("Nhấn 'Chụp ảnh' để bắt đầu quá trình điểm danh.")

# Sử dụng widget camera_input của Streamlit
# Widget này tự động xử lý quyền truy cập camera và trả về một đối tượng File/Bytes
captured_file = st.camera_input("Chụp ảnh điểm danh:")

if captured_file is not None:
    # Đọc bytes của ảnh
    image_bytes = captured_file.getvalue()
    
    with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
        # Gọi hàm xử lý ảnh
        processed_image_np, face_detected, face_locations = detect_and_draw_face(image_bytes)
        
        # Chuyển mảng NumPy về đối tượng Image để hiển thị
        processed_image = Image.fromarray(processed_image_np)
        
    st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
    st.image(processed_image, caption="Ảnh đã được xử lý (Khung xanh/đỏ: Khuôn mặt đã/chưa được phát hiện)", use_column_width=True)

    # Kiểm tra kết quả
    st.markdown("---")
    st.subheader("💡 Kết quả Điểm danh")
    
    if face_detected:
        st.success(f"✅ **Đã phát hiện {len(face_locations)} khuôn mặt.**")
        st.markdown(f"""
        > Hành động tiếp theo: Giả lập quá trình so sánh với dữ liệu dataset.
        > **[Mô phỏng]** So sánh khuôn mặt... Kết quả: **Đã điểm danh thành công!**
        """)
        # Ở đây, bạn sẽ tích hợp logic so sánh khuôn mặt thực tế với dataset của bạn.
        
    else:
        st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
        st.markdown("Vui lòng đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")