# Code to read csv file into Colaboratory:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


linkwhite='https://drive.google.com/open?id=12nRGXBL6u38sCET_x1OmLW6Hlthob-e_'
fluff, id2 = linkwhite.split('=')
print (id2)

downloaded = drive.CreateFile({'id':id2})
downloaded.GetContentFile('data.csv')
df = pd.read_csv('data_.csv')
df.head()
