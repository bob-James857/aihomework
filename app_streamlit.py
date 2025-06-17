import streamlit as st
from ultralytics import YOLO
import PIL.Image
import os
import io

# --- Configuration ---
# Make sure yolo11n.pt is in the same directory as this script.
# For course submission, it's good practice to ensure all required files are together.
MODEL_PATH = 'yolo11n.pt'
DEFAULT_CONFIDENCE = 0.25

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="餐盘智能检测 - 课程作业", # A more formal title for a course assignment
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="auto" # 'auto', 'expanded', 'collapsed'
    # icon="" # Removed icon parameter as per request
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_yolo_model(model_path):
    """
    Loads the YOLO model. Returns the model and None for error, or None and error message.
    """
    if not os.path.exists(model_path):
        return None, f"错误：模型文件 '{model_path}' 未找到。请确保它与 'app_streamlit.py' 在同一目录下。"
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, f"加载模型 '{model_path}' 时出错：{e}。\n请检查 Ultralytics 安装和模型文件完整性。"

# Attempt to load the model
model, model_load_error_message = load_yolo_model(MODEL_PATH)

# --- Main Application Interface ---

st.title("餐盘智能检测系统") # Removed emoji
st.subheader("基于YOLO5的图像识别应用") # Add a subheader for more context

st.markdown("""
本项目是基于YOLO5目标检测模型开发的餐盘识别应用。
上传一张图片，系统将自动检测并标注出图片中的餐盘。
""", unsafe_allow_html=True) # Using unsafe_allow_html=True for potential future HTML styling if needed

# Display model loading status only if there's an error
if model_load_error_message:
    st.error(f"**模型加载失败！**\n\n{model_load_error_message}")
    st.warning("系统核心检测功能不可用。请联系开发者或检查文件路径。")
    # For a course assignment, you might not want to stop the app entirely,
    # but rather disable the functionality. This is handled by the 'if model:' block.
else:
    # No success message here to reduce clutter.
    # The presence of controls implies success.
    pass # No explicit success message shown to keep the main page clean.


# --- Detection Section (only if model loaded successfully) ---
if model:
    st.markdown("---") # Visual separator

    # Slider for confidence threshold
    confidence_slider = st.slider(
        "选择置信度阈值 (Confidence Threshold)",
        min_value=0.05,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE,
        step=0.01,
        format="%.2f", # Format to 2 decimal places
        help="调整此值以控制模型对检测结果的确定程度。值越高，检测结果越可靠，但也可能漏检部分目标。"
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "上传图片进行检测 (支持JPG, JPEG, PNG)", # Removed emoji
        type=["jpg", "jpeg", "png"],
        help="请上传一张包含餐盘的图片。单文件大小限制在 200MB 以内。"
    )

    if uploaded_file is not None:
        try:
            image_input_pil = PIL.Image.open(uploaded_file)
            st.session_state['uploaded_image'] = image_input_pil # Store in session state
        except Exception as e:
            st.error(f"**图片加载失败！** 无法打开上传的图片文件: {e}") # Removed emoji
            st.session_state['uploaded_image'] = None

        if 'uploaded_image' in st.session_state and st.session_state['uploaded_image'] is not None:
            # Display uploaded image
            st.image(st.session_state['uploaded_image'], caption="已上传图片", use_column_width=True) # Removed emoji

            # Button to trigger detection
            col1, col2, col3 = st.columns([1, 2, 1]) # Use columns to center the button
            with col2:
                detect_button = st.button("开始检测 (Detect Plates)", type="primary", use_container_width=True) # Removed emoji

            if detect_button:
                with st.spinner("正在分析图片，请稍候..."): # Removed emoji
                    try:
                        # Perform prediction
                        results = model.predict(source=st.session_state['uploaded_image'], conf=confidence_slider, save=False, verbose=False)

                        # Check if any objects were detected
                        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                            num_detections = len(results[0].boxes)
                            annotated_image_np = results[0].plot() # Returns annotated NumPy array (RGB)
                            annotated_image_pil = PIL.Image.fromarray(annotated_image_np) # Convert back to PIL Image

                            st.success(f"成功检测到 **{num_detections}** 个餐盘！") # Removed emoji
                            st.image(annotated_image_pil, caption="检测结果 (Annotated Image)", use_column_width=True) # Removed emoji
                        else:
                            st.info("未检测到任何餐盘。您可以尝试：\n- 降低置信度阈值\n- 上传背景更简洁的图片") # Removed emoji
                            # Optionally display original image again if no detections
                            # st.image(st.session_state['uploaded_image'], caption="未检测到物体 (No Detections)", use_column_width=True)
                    except Exception as e:
                        st.error(f"**检测过程中发生错误！** 详细信息：{e}") # Removed emoji
    else:
        st.info("请在上方区域上传一张图片，然后点击 '开始检测'。") # Removed emoji

# --- Footer or About Section (Optional) ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>本项目作为课程作业，旨在展示YOLO5在餐盘目标检测中的应用。</p>
    <p><i>开发: [pengsen]</i></p>
</div>
""", unsafe_allow_html=True)

