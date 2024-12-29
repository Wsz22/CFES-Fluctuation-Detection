import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import time

# 使用常量或枚举来管理这些值会更好，以便于维护和理解
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8  # 字体大小
FONT_THICKNESS = 2  # 字体粗细
BOX_THICKNESS = 2  # 边框粗细
BOX_COLOR = (0, 0, 255)  # 边框颜色 BGR
FONT_COLOR = (255, 255, 255)  # 字体颜色
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_AUGMENT = False
DEFAULT_DEVICE = "cpu"


def yoloDetect(uploaded_file, conf, iou, augment, device):
    """
    使用YOLO模型对上传的图片进行目标检测。

    Args:
        uploaded_file (str): 上传的图片文件路径。

    Returns:
        PIL.Image: 检测后的图片，图片中已标出检测到的目标及类别和置信度。

    """
    with st.spinner("Wait for it..."):
        time.sleep(5)
        st.success("Done!")
        img_array = np.array(Image.open(uploaded_file))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        model = YOLO("./model/best.pt")
        results = model(
            img_array, conf=conf, iou=iou, augment=augment, device=device
        )  # 不需要将 img_array 转换为 PIL Image
        result = results[0]

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, FONT_FACE, FONT_SCALE, FONT_THICKNESS
            )

            # 简化标签背景坐标计算
            label_y = y2 + label_height if i % 2 == 0 else y1 + label_height
            cv2.rectangle(
                img_array, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS
            )  # 绘制边界框
            cv2.rectangle(
                img_array,
                (x1, label_y - label_height),
                (x1 + label_width, label_y),
                BOX_COLOR,
                cv2.FILLED,
            )
            cv2.putText(
                img_array,
                label,
                (x1, label_y),
                FONT_FACE,
                FONT_SCALE,
                FONT_COLOR,
                FONT_THICKNESS,
            )
        return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))


def getDevice():
    """
    获取当前可用的设备列表。

    Args:
        无参数。

    Returns:
        list: 包含当前可用设备的列表。列表中包含的设备格式为'cuda:0', 'cuda:1'等，如果CUDA不可用，则列表最后包含'cpu'。

    """
    devices = []
    if torch.cuda.is_available():
        for cuda in range(0, torch.cuda.device_count()):
            devices.append(f"cuda:{cuda}")
    devices.append("cpu")
    st.session_state["devices"] = devices


def reset():
    st.session_state.conf = DEFAULT_CONF
    st.session_state.iou = DEFAULT_IOU
    st.session_state.augment = DEFAULT_AUGMENT
    st.session_state.device = DEFAULT_DEVICE


def playground():
    """
    主函数，用于处理信号波检测应用的主要逻辑。

    Args:
        无

    Returns:
        无

    """
    # 设置页面标题
    st.header("CFES signal wave detection :rocket::rocket::rocket:", divider="gray")
    # 创建两个容器，分别用于显示原始图片和标注图片
    original_img_container = st.container(border=False)
    labeling_img_container = st.container(border=False)
    img_col1, img_col2, img_col3 = st.columns(3)
    # 创建一些session_state，用于存储用户输入的参数
    if "original_img" not in st.session_state:
        st.session_state.original_img = None
    if "conf" not in st.session_state:
        st.session_state.conf = DEFAULT_CONF
    if "iou" not in st.session_state:
        st.session_state.iou = DEFAULT_IOU
    if "augment" not in st.session_state:
        st.session_state.augment = DEFAULT_AUGMENT
    if "device" not in st.session_state:
        st.session_state.device = DEFAULT_DEVICE
    if "devices" not in st.session_state:
        st.session_state.devices = ["cpu"]

    # 创建一个container包含三个列，分别显示三张图片
    with st.container(border=True):
        img_col1.image(os.path.join(".", "img", "img1.png"), "img1")
        img_col2.image(os.path.join(".", "img", "img2.png"), "img2")
        img_col3.image(os.path.join(".", "img", "img3.png"), "img3")
    # 创建一个文件上传组件，允许用户选择一个文件
    user_check = st.selectbox(
        "Please select the image or upload the picture",
        ["uploade img", "img1", "img2", "img3"],
    )
    if user_check == "uploade img":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[
                "bmp",
                "png",
                "jpg",
                "jpeg",
                "dng",
                "tiff",
                "webp",
                "heic",
                "tif",
                "pfm",
            ],
            key="upload_file",
        )
        # 判断是否有文件上传,如果有，添加进session_state
        if uploaded_file is not None:
            st.session_state["original_img"] = uploaded_file
    else:
        if user_check == "img1":
            st.session_state["original_img"] = os.path.join(".", "img", "img1.png")
        elif user_check == "img2":
            st.session_state["original_img"] = os.path.join(".", "img", "img2.png")
        elif user_check == "img3":
            st.session_state["original_img"] = os.path.join(".", "img", "img3.png")
    if st.session_state["original_img"] is not None:
        original_img_container.image(st.session_state["original_img"])

    # 创建一个侧边栏，包含一些参数设置
    with st.sidebar:
        st.header(
            ":rainbow[Arguments setting]:control_knobs:",
            anchor="hello",
            divider="blue",
            help="You can set some model arguments here",
        )

        # 使用 st.session_state 中的值设置滑块初始值
        st.session_state.conf = st.slider(
            "**confidence:**",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=st.session_state.conf,
            label_visibility="visible",
        )
        st.session_state.iou = st.slider(
            "**iou:**",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=st.session_state.iou,
            label_visibility="visible",
        )

        st.markdown("**augment**")
        st.session_state.augment = st.toggle(
            "**augment**", value=st.session_state.augment, label_visibility="hidden"
        )

        st.button(
            "check avliable devices", on_click=getDevice, use_container_width=True
        )
        st.session_state.device = st.selectbox(
            "choose devices", st.session_state["devices"]
        )

        st.button("reset", on_click=reset, use_container_width=True)
        click_status = st.button("detect", use_container_width=True)
        # 判断是否点击了按钮，如果点击了，则进行目标检测
        if click_status:
            if st.session_state["original_img"] is None:
                st.warning("请上传图片", icon="⚠️")
            else:
                output_image = yoloDetect(
                    st.session_state["original_img"],
                    st.session_state["conf"],
                    st.session_state["iou"],
                    st.session_state["augment"],
                    st.session_state["device"],
                )
                labeling_img_container.image(output_image, caption="检测结果")
