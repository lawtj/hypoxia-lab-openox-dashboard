import os
from redcap import Project
import pandas as pd
import numpy as np

def load_project(key):
    api_key = os.environ.get(key)
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    df = project.export_records(format_type='df')
    return df