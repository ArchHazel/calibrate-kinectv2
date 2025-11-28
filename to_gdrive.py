# conda activate gDrive
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

name_on_cloud = 'segmentations.zip'
path_on_local = '/home/Huijun/segmentation.zip'


file = drive.CreateFile({'title': name_on_cloud})
file.SetContentFile(path_on_local)
file.Upload()  
