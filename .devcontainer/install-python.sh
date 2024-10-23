
apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install uv
uv pip install .
uv pip install -r requirements.txt
