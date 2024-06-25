import gdown

# Replace these with your actual Google Drive file IDs
deit_drive_file_id = '1cBGT3951GAXGSVh18RVinC0pDxShkr-N'
davit_drive_file_id = '1xWY-p4jwJTjiXUHCaOz8vA7BmYN9nTC8'

def download_file_from_drive(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_file_from_drive(deit_drive_file_id, 'deit_base_16.pt')
    download_file_from_drive(davit_drive_file_id, 'davit_base_16.pt')
