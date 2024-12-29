import streamlit as st

import streamlit as st


def introduce():
    st.markdown(
        """
            # 恒频电磁信号波动检测系统
        """
    )
    # st.divider()
    st.markdown(
        """
            ## 一、简介
             恒频电磁信号波动检测系统用于检测频谱图上恒频信号的波动，用户上传一个频谱
             图然后点击“detect”即可开始检测，检测结果输出在主屏上。本系统基于 YOLOV8 算法进行
             开发，目标检测速度快，识别精度高，可适配多种硬件平台。
        """
    )
    st.markdown(
        """
            ## 二、安装
    
            **环境及依赖**

            环境要求：python3.10，chrome（推荐）

            **依赖库：**
        """
    )

    st.code(
        """
            streamlit==1.33.0
            streamlit-navigation-bar==3.3.0
            ultralytics==8.3.13
            numpy==1.26.0
            opencv-python-headless
            torch==2.3.0
          """
    )
    st.markdown(
        """
            **启动项目**
            在终端中进入项目主目录，输入 streamlit run app.py 出现下面界面即启动成功。
        """
    )
    st.image("./image.png")
    st.markdown(
        """
            打开浏览器输出界面中的 URL 即可进入项目操作界面
        """
    )
    st.markdown(
        """
            ## 三、快速上手
            ### :one: 上传图像
            进入 Playground 界面，点击中央的"uploade img"下拉框选择要上传的图像。“uploade img”是用户自定义上传。“img1”至“img3”是系统自带的供用户测试的图片。若选择“uploade img”模式，用户点击下方的上传栏从本地选择图片即可完成上传
            ### 2️⃣ 开始检测
            点击屏幕左上角 ▶️ 按钮唤出隐藏框，设置好参数后（参数设置见 3）点击“detect“开始检测，界面显示“Wait for it...“说明系统正在检测，当图标变成”Done!“后检测完成，结果显示在主界面上。
            ### :three: 参数设置
            各种参数含义见下表                                                                                                                                                                                                          
    """
    )
    st.image("./parameterList.png", width=1000)
    st.image("./demonstrate.gif")
    st.markdown(
        """
            ## 四、联系我们
            Github 地址：https://github.com/Wsz22/cfes

            Email：wusizheng@yeah.net
        """
    )
