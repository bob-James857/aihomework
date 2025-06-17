import streamlit as st
from ultralytics import YOLO
import PIL.Image
import os
import logging
import io # 用于处理Streamlit上传的文件

# --- 配置 ---
MODEL_PATH = 'yolo11n.pt'  # 请确保这是您模型的准确路径
DEFAULT_CONFIDENCE = 0.25

# --- 日志记录 (Streamlit中可以直接使用st.write, st.info, st.error等，logging也可用于后台) ---
# logging.basicConfig(level=logging.INFO) # Streamlit运行时，logging输出到控制台
# logger = logging.getLogger(__name__)
# 为了在Streamlit界面上更直观，我们将主要使用st的打印功能

# --- 模型加载 ---
# 使用 Streamlit 的缓存功能，确保模型只加载一次
@st.cache_resource # 对于像模型这样的复杂对象，使用 st.cache_resource
def load_yolo_model(model_path):
    """
    加载YOLO模型。如果加载失败返回None和错误信息。
    """
    # logger.info(f"Streamlit: 检查模型文件路径: {model_path}")
    if not os.path.exists(model_path):
        error_msg = f"错误: 模型文件 {model_path} 未找到。请检查路径。"
        # logger.error(f"Streamlit: {error_msg}")
        return None, error_msg
    try:
        # logger.info(f"Streamlit: 尝试从 {model_path} 加载 YOLO 模型...")
        model = YOLO(model_path)  # 加载您的 YOLO 模型
        # logger.info(f"Streamlit: 模型 {model_path} 加载成功。")
        return model, None
    except Exception as e:
        error_msg = f"加载模型 {model_path} 时出错: {e}。请确保 Ultralytics 已正确安装，模型文件有效。"
        # logger.error(f"Streamlit: {error_msg}")
        return None, error_msg

# 尝试加载模型
model, model_load_error_message = load_yolo_model(MODEL_PATH)

# --- Streamlit 界面 ---
st.title("YOLO 餐盘检测 (Streamlit Demo)")

st.markdown(f"""
上传一张包含餐盘的图片，模型将会检测并标注出餐盘的位置。
**模型路径:** `{MODEL_PATH}`
""")

if model_load_error_message:
    st.error(f"模型加载失败！错误信息：{model_load_error_message}")
    st.warning("由于模型未能加载，检测功能将不可用。请检查服务器控制台日志以获取详细信息。")
    # st.stop() # 如果希望在模型加载失败时完全停止应用
else:
    if model is not None:
        st.success(f"模型 {MODEL_PATH} 加载成功！可以开始检测了。")
    else: # 理论上不应到这里，因为load_yolo_model会返回错误信息
        st.error("未知错误导致模型未能加载。")


# 仅当模型成功加载时才显示UI控件
if model:
    # --- 输入控件 ---
    confidence_slider = st.slider(
        "置信度阈值 (Confidence Threshold)",
        min_value=0.05,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE,
        step=0.01
    )

    uploaded_file = st.file_uploader(
        "上传图片 (Upload Image)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # 将上传的文件转换为PIL Image对象
        try:
            image_input_pil = PIL.Image.open(uploaded_file)
        except Exception as e:
            st.error(f"无法打开上传的图片文件: {e}")
            image_input_pil = None

        if image_input_pil:
            st.image(image_input_pil, caption="您上传的图片 (Uploaded Image)", use_column_width=True)

            # 添加一个按钮来触发检测，避免每次调整滑块都重新检测
            if st.button("开始检测 (Detect Plates)"):
                with st.spinner("正在检测中..."): # 显示加载状态
                    # logger.info(f"Streamlit: 接收到图片，使用置信度 {confidence_slider} 进行餐盘检测...")
                    try:
                        # 使用 YOLO 模型进行预测
                        results = model.predict(source=image_input_pil, conf=confidence_slider, save=False, verbose=False)

                        # 检查是否有检测结果
                        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                            annotated_image_np = results[0].plot() # 返回一个带有标注的 NumPy 数组 (RGB格式)
                            annotated_image_pil = PIL.Image.fromarray(annotated_image_np) # 转换回 PIL Image
                            # logger.info("Streamlit: 检测完成，显示标注后的图片。")
                            st.image(annotated_image_pil, caption="检测结果 (Detection Result)", use_column_width=True)
                        else:
                            # logger.info("Streamlit: 未检测到任何物体，或结果为空。显示原始图片。")
                            st.info("未检测到任何物体，或检测结果为空。")
                            # st.image(image_input_pil, caption="未检测到物体 (No Detections)", use_column_width=True) # 可选：再次显示原图
                    except Exception as e:
                        # logger.error(f"Streamlit: 在检测过程中发生错误: {e}")
                        st.error(f"处理图像时发生严重错误: {e}")
    else:
        st.info("请上传一张图片进行检测。")

# --- 如何运行 Streamlit 应用的说明 ---
st.sidebar.header("如何运行")
st.sidebar.markdown

