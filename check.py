# check.py (Phần logic đã thay đổi trong hàm main_app)

# ----------------------------------------------------------------------
#                             GIAO DIỆN CHÍNH (main_app)
# ----------------------------------------------------------------------

def main_app(credentials):
    """
    Hàm chứa toàn bộ logic giao diện Streamlit.
    """
    
    # ... (Phần tải Dataset & Checklist giữ nguyên)
    # 1. Tải Dataset & Checklist
    from config import GDRIVE_DATASET_FOLDER_ID, GDRIVE_CHECKLIST_ID
    from config import download_dataset_folder_real
    
    # Tải Folder Dataset (REAL) - Truyền CREDENTIALS vào tham số _credentials
    dataset_ready = download_dataset_folder_real(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, credentials) 
    # Tải Checklist (XLSX) - Truyền CREDENTIALS vào tham số _credentials
    # KHÔNG CÓ CACHE: Luôn tải bản mới nhất từ Drive
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

    st.info(f"Checklist có {len(checklist_df)} người.")

    # 2. Chọn Buổi Học (Dropdown)
    attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

    if not attendance_cols:
         st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
         return
         
    # --- THAY ĐỔI: Thêm một giá trị mặc định (placeholder) vào đầu danh sách ---
    placeholder_option = "--- Chọn Buổi Điểm Danh ---"
    options_with_placeholder = [placeholder_option] + attendance_cols

    selected_session = st.selectbox(
        "1️⃣ **Chọn Buổi Điểm Danh**", 
        options_with_placeholder, 
        index=0, # Chọn placeholder làm mặc định
        help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
    )
    
    st.markdown("---")

    # --- THAY ĐỔI: Kiểm tra xem buổi học đã được chọn hợp lệ chưa ---
    if selected_session == placeholder_option:
        st.info("💡 Vui lòng chọn một **Buổi Điểm Danh** để bắt đầu.")
        # Hiển thị checklist ngay cả khi chưa chọn buổi
        
        # 3. Trạng thái Checklist Hiện tại
        st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
        if CHECKLIST_SESSION_KEY in st.session_state:
            current_df = st.session_state[CHECKLIST_SESSION_KEY]
            st.dataframe(current_df)
            
            # (Phần nút tải Excel giữ nguyên)
            # 1. Tạo file Excel trong bộ nhớ (sử dụng io.BytesIO)
            output = io.BytesIO()
            # Lưu DataFrame vào buffer, bỏ index
            current_df.to_excel(output, index=False, sheet_name='Checklist_Cap_Nhat')
            excel_data = output.getvalue()
            
            # 2. Hiển thị nút tải về
            st.download_button(
                label="⬇️ Tải file Excel Checklist đã cập nhật",
                data=excel_data,
                file_name="Checklist_DiemDanh_CapNhat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Tải về file Excel (XLSX) chứa dữ liệu điểm danh mới nhất trong phiên làm việc hiện tại."
            )
            
        return # Thoát khỏi hàm nếu chưa chọn buổi

    # Nếu đã chọn buổi hợp lệ
    st.success(f"Đang điểm danh cho: **{selected_session}**")

    # 4. Chụp Ảnh và Xử Lý (CHỈ KHI ĐÃ CHỌN BUỔI HỢP LỆ)
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

        # ... (Phần logic kết quả điểm danh giữ nguyên)
        if stt_match and distance is not None: # Đảm bảo cả stt_match và distance đều có giá trị
            st.balloons()
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **STT trùng khớp:** **{stt_match}**
            * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
            """)
            # Cập nhật checklist (KHÔNG Ghi ngược lên Drive, chỉ cập nhật session state)
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
    # 5. Trạng thái Checklist Hiện tại
    st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
    if CHECKLIST_SESSION_KEY in st.session_state:
        current_df = st.session_state[CHECKLIST_SESSION_KEY]
        st.dataframe(current_df)
        
        # --- BỔ SUNG NÚT TẢI VỀ FILE EXCEL ---
        # 1. Tạo file Excel trong bộ nhớ (sử dụng io.BytesIO)
        output = io.BytesIO()
        # Lưu DataFrame vào buffer, bỏ index
        current_df.to_excel(output, index=False, sheet_name='Checklist_Cap_Nhat')
        excel_data = output.getvalue()
        
        # 2. Hiển thị nút tải về
        st.download_button(
            label="⬇️ Tải file Excel Checklist đã cập nhật",
            data=excel_data,
            file_name="Checklist_DiemDanh_CapNhat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Tải về file Excel (XLSX) chứa dữ liệu điểm danh mới nhất trong phiên làm việc hiện tại."
        )
        # --------------------------------------
