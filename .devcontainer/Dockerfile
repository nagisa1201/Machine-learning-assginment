# 选择 ubuntu:20.04 原生镜像，兼容旧版本依赖库
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# ==========================================
# 用户和用户组定义
# ==========================================
ARG USERNAME=Nagisa
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG GROUP_NAME=wheel

RUN set -eux; \
    if getent passwd 1000 > /dev/null; then \
    existing_user=$(getent passwd 1000 | cut -d: -f1); \
    userdel -r "$existing_user" 2>/dev/null || true; \
    fi; \
    if getent group 1000 > /dev/null; then \
    existing_group=$(getent group 1000 | cut -d: -f1); \
    if ! getent passwd | awk -F: '{print $1":"$4}' | grep -q ":1000$"; then \
    groupdel "$existing_group" 2>/dev/null || true; \
    fi; \
    fi; \
    groupadd --gid $USER_GID ${GROUP_NAME} && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    echo "$USERNAME:password" | chpasswd && \
    usermod -aG ${GROUP_NAME} $USERNAME && \
    # 安装 sudo 以供后续 make install 使用
    apt-get update && apt-get install -y sudo && \
    echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown $USERNAME:$GROUP_NAME /home/$USERNAME/.bashrc

# ==========================================
# 安装系统依赖包
# ==========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl nano xautomation \
    libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev \
    libxrender-dev libxrandr-dev libpng-dev \
    python3-dev zlib1g-dev libjpeg-dev cmake swig \
    python3-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb sed \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# 3. 安装 Miniconda (全局安装以供用户使用)
# ==========================================
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && chown -R $USERNAME:$GROUP_NAME /opt/conda

# 切换到您的专属用户
USER $USERNAME
WORKDIR /home/$USERNAME

# 配置 Conda 环境变量
ENV PATH="/opt/conda/bin:$PATH"

# 创建 conda 虚拟环境 (使用 Python 3.6 以匹配 cp36 的 PyTorch wheel 包)
RUN conda create -n ddpg python=3.6 -y && \
    conda clean -ya

# 设置终端以激活 Conda 虚拟环境
SHELL ["/opt/conda/bin/conda", "run", "-n", "ddpg", "/bin/bash", "-c"]

# ==========================================
# 下载并搭建 TORCS 和 Gym-TORCS 环境
# ==========================================
# 步骤 一 & 四：下载代码库
RUN git clone https://github.com/ugo-nama-kun/gym_torcs.git && \
    git clone https://github.com/jastfkjg/DDPG_Torcs_PyTorch

# 步骤 二(4) & (5)：修改 geometry.cpp 源码中的 isnan 为 std::isnan
WORKDIR /home/$USERNAME/gym_torcs/vtorcs-RL-color
RUN sed -i 's/if (isnan(r))/if (std::isnan(r))/g' src/drivers/olethros/geometry.cpp

# 步骤 二(6)：编译并安装 TORCS (由于涉及系统目录，需要使用 sudo)
RUN ./configure && \
    make -j$(nproc) && \
    sudo make install && \
    sudo make datainstall

# ==========================================
# 配置 Python 依赖
# ==========================================
# 安装 numpy, gym (步骤二 4.3)
RUN pip install numpy gym

# 步骤 四(2) & (3)：安装指定版本的 PyTorch 和 Torchvision
WORKDIR /home/$USERNAME/DDPG_Torcs_PyTorch
RUN pip install https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install torchvision==0.2.1

# ==========================================
# 收尾工作
# ==========================================
# 写入 bashrc 以确保登入容器时默认激活 ddpg 环境
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc && \
    echo "conda activate ddpg" >> /home/$USERNAME/.bashrc

# 重置工作目录到主目录
WORKDIR /home/$USERNAME

# 容器启动时默认进入 bash
CMD ["/bin/bash"]