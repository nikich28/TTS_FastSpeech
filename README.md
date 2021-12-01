### TextToSpeech with FastSpeech

Commands for training in google colab:
~~~
!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xjf LJSpeech-1.1.tar.bz2

!git clone https://github.com/nikich28/TTS_FastSpeech.git

%cd TTS_FastSpeech

!pip install -r ./requirements.txt
!pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

!git clone https://github.com/NVIDIA/waveglow.git
!pip install googledrivedownloader

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
    dest_path='./waveglow_256channels_universal_v5.pt'
)

!python3 train.py
