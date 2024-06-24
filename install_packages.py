import subprocess

def install_packages():
    packages = [
        "pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu118",
        "pip install datasets matplotlib pandas pillow torcheval torchtnt tqdm",
        "pip install timm==0.9.12",
        "pip install cjm_pandas_utils cjm_pil_utils cjm_pytorch_utils"
    ]
    
    for package in packages:
        subprocess.run(package, shell=True, check=True)

if __name__ == "__main__":
    install_packages()
