# 选择 ubuntu:20.04 原生镜像
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# ==========================================
# 1. 安装系统级依赖包
# ==========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl nano xautomation sudo \
    libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev \
    libxrender-dev libxrandr-dev libpng-dev \
    python3-dev zlib1g-dev libjpeg-dev cmake swig \
    python3-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb sed ca-certificates \
    fonts-wqy-zenhei \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==========================================
# 2. 编译并安装 TORCS 底层引擎
# ==========================================
# 将宿主机上的 gym_torcs 拷贝进来编译 (依赖于 compose 中的 context: ..)
COPY gym_torcs /opt/gym_torcs_build

# 编译并安装模拟器到系统目录 /usr/local
RUN cd /opt/gym_torcs_build/vtorcs-RL-color && \
    mkdir -p /usr/local/bin /usr/local/lib /usr/local/include /usr/local/share/games/torcs && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    make datainstall && \
    # 编译完成后删除临时源码，运行时将使用你外部挂载的实时代码
    rm -rf /opt/gym_torcs_build

# ==========================================
# 3. 安装 Conda 并配置系统用户
# ==========================================
ARG USERNAME=Nagisa
ARG USER_UID=1000
ARG USER_GID=1000

RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    # 用户创建与 Sudo 提权
    groupadd --gid $USER_GID wheel && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    echo "$USERNAME:password" | chpasswd && \
    echo "%wheel ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R $USERNAME:wheel /home/$USERNAME /opt/conda

# ==========================================
# 4. 配置 Python 环境 
# ==========================================
USER $USERNAME
WORKDIR /home/$USERNAME
ENV PATH="/opt/conda/bin:$PATH"

# 创建虚拟环境 -> 安装依赖 -> 写入 bashrc 
RUN conda create -n ddpg -c conda-forge --override-channels python=3.6 -y && \
    conda clean -ya && \
    /opt/conda/bin/conda run -n ddpg pip install numpy gym pandas scikit-learn matplotlib seaborn && \
    /opt/conda/bin/conda run -n ddpg pip install https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/conda/bin/conda run -n ddpg pip install torchvision==0.2.1 && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc && \
    echo "conda activate ddpg" >> /home/$USERNAME/.bashrc

# ==========================================
# 容器启动命令
# ==========================================
CMD ["/bin/bash"]