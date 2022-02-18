import streamlit as st
import os
import zipfile
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files


# globals

new_folder: str = os.path.join(os.getcwd(), r"new_folder\\")


def extract(zip_file):
    with zipfile.ZipFile(new_folder + zip_file.name, "r") as zip_ref:
        zip_ref.extractall(new_folder)
        st.success("File Unzipped")


def app():
    st.title('Upload data')

    st.write('File conversion from DICOM to NRRD.')

    st.write('Feature extraction is presented below.')

    zip_file = st.file_uploader("Upload zip of patient",
                                type=["zip"])

    if zip_file is not None:
        # TO See details
        file_details = {"filename": zip_file.name, "filetype": zip_file.type,
                        "filesize": zip_file.size}
        st.write(file_details)
        # Saving upload
        with open(os.path.join(new_folder, zip_file.name), "wb") as f:
            f.write(zip_file.getbuffer())

        st.success("File Saved")
        extract(zip_file)


