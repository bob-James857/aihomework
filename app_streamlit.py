import streamlit as st
from ultralytics import YOLO
import PIL.Image
import os
import io

# --- 配置 ---
MODEL_PATH = 'yolo11n.pt'
DEFAULT_CONFIDENCE = 0.25

# --- 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    page_title="餐盘智能检测 - 课程作业", # 页面标题
    layout="centered", # 布局居中
    initial_sidebar_state="auto" # 侧边栏状态自动
)

# --- CSS / 美化代码注入 (卡片式设计) ---
st.markdown("""
<style>
/* 全局样式调整 */
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    color: #26272E; /* 深灰色文本 */
    background-color: #F0F2F6; /* 统一的浅灰色背景 */
}

/* 主容器的内边距，让内容不至于紧贴边缘 */
.stApp {
    padding: 20px;
}

/* 自定义卡片样式 */
.st-card {
    background-color: #FFFFFF; /* 卡片背景白色 */
    border-radius: 12px; /* 圆角边框 */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* 柔和的阴影效果 */
    padding: 25px; /* 内边距 */
    margin-bottom: 25px; /* 卡片之间间距 */
    border: 1px solid #E0E0E0; /* 轻微边框 */
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* 悬停动画 */
}

.st-card:hover {
    transform: translateY(-3px); /* 向上轻微浮动 */
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12); /* 阴影增强 */
}

/* 主标题美化 */
h1 {
    color: #336699; /* 蓝色系标题 */
    font-weight: 700;
    margin-bottom: 1em; /* 增加底部间距 */
    text-align: center; /* 标题居中 */
}

/* 副标题美化 */
h2 {
    color: #4CAF50; /* 绿色系副标题 */
    font-weight: 600;
    margin-top: 1.5em;
    margin-bottom: 1em;
    text-align: center; /* 副标题居中 */
}

/* 模块小标题 */
h4 {
    color: #444;
    font-size: 1.3em;
    font-weight: 600;
    margin-bottom: 1em;
    border-bottom: 1px solid #F0F0F0; /* 底部细线 */
    padding-bottom: 8px;
}

/* 文本段落 */
p {
    line-height: 1.7;
    letter-spacing: 0.3px;
    color: #4A4A4A;
}

/* 分隔线美化 */
hr {
    border-top: 1px dashed #D0D0D0;
    margin-top: 2em;
    margin-bottom: 2em;
}

/* 按钮美化 (Primary Button) */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px; /* 更圆润的按钮 */
    padding: 0.8em 1.5em;
    font-weight: 600;
    border: none;
    transition: background-color 0.2s, transform 0.2s;
    font-size: 1.1em; /* 按钮文字稍大 */
}
.stButton>button:hover {
    background-color: #45A049;
    transform: translateY(-2px); /* 悬停时轻微上浮 */
}

/* 文件上传区域 */
.stFileUploader {
    padding: 20px;
    border: 2px dashed #B0B0B0; /* 更明显的虚线边框 */
    border-radius: 10px;
    background-color: #FAFAFA;
    transition: all 0.2s ease-in-out;
}
.stFileUploader:hover {
    border-color: #4CAF50;
    box-shadow: 0 0 8px rgba(76, 175, 80, 0.4);
}

/* 信息提示框 (st.info, st.success, st.error, st.warning) */
.stAlert {
    border-radius: 10px;
    padding: 18px 25px;
    margin-top: 1.5em;
    margin-bottom: 1.5em;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1);
}

/* 页脚样式 */
.footer {
    text-align: center;
    color: #888;
    font-size: 0.85em;
    margin-top: 4em;
    padding-top: 2em;
    border-top: 1px solid #E5E5E5;
}

</style>
""", unsafe_allow_html=True)


# --- 模型加载 (缓存资源) ---
@st.cache_resource
def load_yolo_model(model_path):
    """
    加载YOLO模型。
    """
    if not os.path.exists(model_path):
        return None, f"错误：模型文件 '{model_path}' 未找到。请检查路径。"
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, f"模型加载失败: {e}。请检查Ultralytics安装和模型文件有效性。"

# 尝试加载模型
model, model_load_error_message = load_yolo_model(MODEL_PATH)

# --- 主应用界面 ---

st.title("餐盘智能检测系统")
st.subheader("基于YOLOv5的图像识别应用") # 更新为YOLOv5描述

st.markdown("""
本项目实现了基于YOLOv5目标检测模型的餐盘识别功能。
用户可上传图片进行检测，系统将自动标注识别到的餐盘目标。
""")

# 模型加载状态提示
if model_load_error_message:
    st.error(f"**模型加载失败！** 错误详情：\n\n{model_load_error_message}")
    st.warning("检测功能不可用。请检查模型文件或部署环境。")
else:
    pass # 模型加载成功时不显示额外信息

# --- 检测功能区 ---
if model:
    # --- 参数设置卡片 ---
    st.markdown('<div class="st-card">', unsafe_allow_html=True) # 卡片开始
    st.markdown("#### 参数设置")
    confidence_slider = st.slider(
        "置信度阈值",
        min_value=0.05,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE,
        step=0.01,
        format="%.2f",
        help="调整目标检测的置信度阈值。"
    )
    st.markdown("</div>", unsafe_allow_html=True) # 卡片结束
    
    st.markdown("<br>", unsafe_allow_html=True) # 增加卡片间距

    # --- 图片上传与检测卡片 ---
    st.markdown('<div class="st-card">', unsafe_allow_html=True) # 卡片开始
    st.markdown("#### 图片上传与检测")
    uploaded_file = st.file_uploader(
        "上传图片",
        type=["jpg", "jpeg", "png"],
        help="上传一张包含餐盘的图片进行检测。文件大小上限 200MB。"
    )

    if uploaded_file is not None:
        try:
            image_input_pil = PIL.Image.open(uploaded_file)
            st.session_state['uploaded_image'] = image_input_pil
        except Exception as e:
            st.error(f"**图片加载失败！** 错误详情: {e}")
            st.session_state['uploaded_image'] = None

        if 'uploaded_image' in st.session_state and st.session_state['uploaded_image'] is not None:
            # 显示上传图片
            st.image(st.session_state['uploaded_image'], caption="上传图片", use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True) # 增加垂直间距

            # 检测按钮
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                detect_button = st.button("开始检测", type="primary", use_container_width=True)

            if detect_button:
                st.markdown("</div>", unsafe_allow_html=True) # 图片上传卡片结束
                
                # --- 检测结果卡片 ---
                st.markdown("<br>", unsafe_allow_html=True) # 增加卡片间距
                st.markdown('<div class="st-card">', unsafe_allow_html=True) # 新卡片开始
                st.markdown("#### 检测结果")
                with st.spinner("正在执行检测..."):
                    try:
                        results = model.predict(source=st.session_state['uploaded_image'], conf=confidence_slider, save=False, verbose=False)

                        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                            num_detections = len(results[0].boxes)
                            annotated_image_np = results[0].plot()
                            annotated_image_pil = PIL.Image.fromarray(annotated_image_np)

                            st.success(f"检测到 **{num_detections}** 个目标。")
                            st.image(annotated_image_pil, caption="检测结果", use_column_width=True)
                        else:
                            st.info("未检测到目标。")
                    except Exception as e:
                        st.error(f"**检测过程中发生错误！** 错误详情：{e}")
                st.markdown("</div>", unsafe_allow_html=True) # 检测结果卡片结束
            else: # 如果没有点击检测按钮，且图片已上传
                st.markdown("</div>", unsafe_allow_html=True) # 关闭图片上传卡片

    else: # 没有上传文件时
        st.info("请上传图片以开始检测。")
        st.markdown("</div>", unsafe_allow_html=True) # 关闭图片上传卡片
        
# --- 底部信息 ---
st.markdown("""
<div class="footer">
    <p>本项目为[人工智能]课程的[实验报告]。</p>
    <p>开发: pengsen</p>
</div>
""", unsafe_allow_html=True)
