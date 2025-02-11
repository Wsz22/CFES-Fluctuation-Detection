import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import time
import pandas as pd

# 常量定义
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_THICKNESS = 2
BOX_COLOR = (0, 0, 255)
FONT_COLOR = (255, 255, 255)
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_AUGMENT = False
DEFAULT_DEVICE = "CPU"


def yoloDetect(uploaded_file, conf, iou, augment, device):
    """目标检测并返回标注图像和检测数据"""
    with st.spinner("检测中，请稍候..."):
        try:
            start_time = time.time()

            # 图像预处理
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 加载模型
            model = YOLO("./model/best.pt")

            # 执行推理
            results = model(
                img_array,
                conf=conf,
                iou=iou,
                augment=augment,
                device=device,
                verbose=False
            )
            result = results[0]

            detection_data = []
            wave_counter = 1

            # 处理每个检测框
            for box in result.boxes:
                # 坐标解析
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 特征计算
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                # 获取检测信息
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                # 收集数据
                detection_data.append({
                    "id": wave_counter,
                    "name": class_name,
                    "中心X": center_x,
                    "中心Y": center_y,
                    "长": width,
                    "宽": height,
                    "置信度": confidence
                })

                # 可视化标注
                label = f"{class_name}{wave_counter}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, FONT_FACE, FONT_SCALE, FONT_THICKNESS
                )

                # 动态标签位置
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

                # 绘制检测框
                cv2.rectangle(
                    img_array,
                    (x1, y1),
                    (x2, y2),
                    BOX_COLOR,
                    BOX_THICKNESS
                )

                # 绘制标签背景
                cv2.rectangle(
                    img_array,
                    (x1, label_y - label_height),
                    (x1 + label_width, label_y),
                    BOX_COLOR,
                    cv2.FILLED,
                )

                # 添加文本
                cv2.putText(
                    img_array,
                    label,
                    (x1, label_y),
                    FONT_FACE,
                    FONT_SCALE,
                    FONT_COLOR,
                    FONT_THICKNESS,
                )

                wave_counter += 1

            # 转换颜色空间
            output_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

            # 性能统计
            process_time = time.time() - start_time
            st.toast(f"检测完成！耗时 {process_time:.2f}秒", icon="✅")

            return output_img, detection_data

        except Exception as e:
            st.error(f"检测失败: {str(e)}")
            return None, []


def get_available_devices():
    """获取可用计算设备列表"""
    devices = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    devices.append("cpu")
    return devices



def playground():
    """
    信号波形检测主界面（完整可运行版本）
    """
    # 初始化session状态
    defaults = {
        "original_img": None,
        "conf": DEFAULT_CONF,
        "iou": DEFAULT_IOU,
        "augment": DEFAULT_AUGMENT,
        "devices": get_available_devices(),
        "device": get_available_devices()[0],
        "detection_data": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 页面标题和分隔线
    st.header("CFES Signal Wave Detection System 🚀", divider="rainbow")

    # ========== 侧边栏设置 ==========
    with st.sidebar:
        st.header("⚙️ 参数设置", divider="blue")

        # 置信度和IOU设置
        st.session_state.conf = st.slider(
            "置信度阈值", 0.0, 1.0, DEFAULT_CONF, 0.01,
            help="过滤低置信度检测结果的阈值"
        )
        st.session_state.iou = st.slider(
            "IoU阈值", 0.0, 1.0, DEFAULT_IOU, 0.01,
            help="非极大值抑制的交并比阈值"
        )

        # 高级设置折叠区域
        with st.expander("高级设置", expanded=False):
            st.session_state.augment = st.checkbox(
                "TTA增强", DEFAULT_AUGMENT,
                help="启用测试时数据增强（可能提高精度但降低速度）"
            )

            # 设备选择组件
            device_info = []
            for dev in st.session_state.devices:
                if dev.startswith("cuda"):
                    idx = int(dev.split(":")[1])
                    prop = torch.cuda.get_device_properties(idx)
                    info = f"{prop.name} | 显存：{prop.total_memory / 1024 ** 3:.1f}GB"
                else:
                    info = "CPU"
                device_info.append(info)

            st.session_state.device = st.selectbox(
                "计算设备",
                options=st.session_state.devices,
                format_func=lambda x: device_info[st.session_state.devices.index(x)],
                index=0
            )

            # 显存监控
            if st.session_state.device.startswith("cuda"):
                device_id = int(st.session_state.device.split(":")[1])
                mem_alloc = torch.cuda.memory_allocated(device_id) / 1024 ** 3
                mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
                st.metric("显存使用", f"{mem_alloc:.2f}/{mem_total:.2f} GB")

        # 操作按钮
        st.divider()
        detect_clicked = st.button("🔍 开始检测", use_container_width=True)

    # ========== 主界面内容 ==========
    # 样本图片展示区
    with st.container(border=True):
        cols = st.columns(3)
        sample_images = {
            "img1": "img/img1.png",
            "img2": "img/img2.png",
            "img3": "img/img3.png"
        }
        for col, (name, path) in zip(cols, sample_images.items()):
            with col:
                st.image(path, caption=name, use_column_width=True)

    # 图片选择/上传区
    selected_img = st.radio(
        "选择输入来源：",
        ["样本图片", "上传图片"],
        horizontal=True,
        index=0
    )

    if selected_img == "上传图片":
        uploaded_file = st.file_uploader(
            "上传检测图片",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="uploader"
        )
        if uploaded_file:
            st.session_state.original_img = uploaded_file
    else:
        img_choice = st.selectbox("选择样本图片", list(sample_images.keys()))
        st.session_state.original_img = sample_images[img_choice]

    # 显示原始图片
    if st.session_state.original_img:
        if isinstance(st.session_state.original_img, str):  # 样本图片路径
            img = Image.open(st.session_state.original_img)
        else:  # 上传的文件对象
            img = Image.open(st.session_state.original_img)

        with st.expander("原始图片预览", expanded=True):
            st.image(img, caption="原始图片", use_column_width=True)

    # 执行检测并显示结果
    if detect_clicked:
        if not st.session_state.original_img:
            st.warning("请先选择或上传图片！")
            st.stop()

        try:
            # 执行检测
            start_time = time.time()
            result_img, detection_data = yoloDetect(
                st.session_state.original_img,
                st.session_state.conf,
                st.session_state.iou,
                st.session_state.augment,
                st.session_state.device
            )
            process_time = time.time() - start_time

            # 显示处理结果
            with st.container(border=True):
                cols = st.columns([0.7, 0.3])
                with cols[0]:
                    st.image(result_img, caption="检测结果", use_column_width=True)
                with cols[1]:
                    st.metric("处理耗时", f"{process_time:.2f}s")
                    st.metric("检测到目标数", len(detection_data))

            # 显示数据表格
            st.subheader("检测数据详情", divider="grey")
            df = pd.DataFrame(detection_data)
            st.dataframe(
                df,
                column_config={
                    "center_x": st.column_config.NumberColumn("X坐标", format="%d px"),
                    "center_y": st.column_config.NumberColumn("Y坐标", format="%d px"),
                    "width": st.column_config.NumberColumn("宽度", format="%d px"),
                    "height": st.column_config.NumberColumn("高度", format="%d px"),
                    "confidence": st.column_config.NumberColumn("置信度", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"检测过程中发生错误：{str(e)}")
            st.exception(e)
