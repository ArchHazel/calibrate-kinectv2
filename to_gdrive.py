from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file = drive.CreateFile({'title': 'bin_files.zip'})
file.SetContentFile('bin_files.zip')
file.Upload()  
