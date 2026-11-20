"""
create_demo_dicom.py
====================
Generates standard clinical DICOM (.dcm) files from reference MRI scans with
realistic DICOM tags (Siemens MAGNETOM, GE SIGNA, Philips Achieva), proper
pixel data encodings, and clinical metadata for testing native DICOM workflows.
"""

from pathlib import Path
import datetime
import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

DEMO_CASES = [
    {
        "id": "case_01_glioma",
        "name": "Case 01: High-Grade Glioma (Glioblastoma)",
        "source_jpg": "/home/sidd/Downloads/tumor/dataset/Testing/glioma/Te-gl_100.jpg",
        "out_dcm": "glioma_high_grade.dcm",
        "patient_id": "MR-2026-GL01",
        "patient_age": "056Y",
        "patient_sex": "M",
        "series_desc": "Axial T1+Gd Contrast Neuro",
        "manufacturer": "SIEMENS",
        "model": "MAGNETOM Prisma 3.0T",
        "magnetic_field": 3.0,
        "slice_thickness": 2.5,
        "pixel_spacing": [0.65, 0.65],
        "tr": 680.0,
        "te": 12.0,
    },
    {
        "id": "case_02_meningioma",
        "name": "Case 02: Parasagittal Dural Meningioma",
        "source_jpg": "/home/sidd/Downloads/tumor/dataset/Testing/meningioma/Te-aug-me_100.jpg",
        "out_dcm": "meningioma_parasagittal.dcm",
        "patient_id": "MR-2026-ME02",
        "patient_age": "061Y",
        "patient_sex": "F",
        "series_desc": "Axial T1 Post-Contrast",
        "manufacturer": "GE Healthcare",
        "model": "SIGNA Premier 3.0T",
        "magnetic_field": 3.0,
        "slice_thickness": 3.0,
        "pixel_spacing": [0.72, 0.72],
        "tr": 750.0,
        "te": 14.0,
    },
    {
        "id": "case_03_pituitary",
        "name": "Case 03: Pituitary Sellar Macroadenoma",
        "source_jpg": "/home/sidd/Downloads/tumor/dataset/Testing/pituitary/Te-pi_100.jpg",
        "out_dcm": "pituitary_sellar_adenoma.dcm",
        "patient_id": "MR-2026-PI03",
        "patient_age": "039Y",
        "patient_sex": "F",
        "series_desc": "Coronal/Axial Sellar Dedicated T1+C",
        "manufacturer": "Philips Medical Systems",
        "model": "Achieva 1.5T",
        "magnetic_field": 1.5,
        "slice_thickness": 2.0,
        "pixel_spacing": [0.55, 0.55],
        "tr": 520.0,
        "te": 10.0,
    },
    {
        "id": "case_04_normal",
        "name": "Case 04: Normal Unremarkable Control",
        "source_jpg": "/home/sidd/Downloads/tumor/dataset/Testing/notumor/Te-no_100.jpg",
        "out_dcm": "normal_control_brain.dcm",
        "patient_id": "MR-2026-NC04",
        "patient_age": "031Y",
        "patient_sex": "M",
        "series_desc": "Axial Brain T1 Spin Echo",
        "manufacturer": "SIEMENS",
        "model": "MAGNETOM Vida 3.0T",
        "magnetic_field": 3.0,
        "slice_thickness": 3.0,
        "pixel_spacing": [0.75, 0.75],
        "tr": 600.0,
        "te": 11.0,
    },
]


def convert_jpg_to_dicom(case_cfg: dict, out_dir: Path):
    source_path = Path(case_cfg["source_jpg"])
    if not source_path.exists():
        print(f"Warning: Source {source_path} does not exist, creating placeholder")
        gray_img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(gray_img, (128, 128), 80, 180, -1)
    else:
        img = cv2.imread(str(source_path))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape

    # File meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out_file = out_dir / case_cfg["out_dcm"]
    ds = FileDataset(str(out_file), {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient & Study attributes
    ds.PatientName = case_cfg["patient_id"]
    ds.PatientID = case_cfg["patient_id"]
    ds.PatientAge = case_cfg["patient_age"]
    ds.PatientSex = case_cfg["patient_sex"]
    ds.Modality = "MR"
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "Brain MRI Clinical Evaluation"
    ds.SeriesDescription = case_cfg["series_desc"]
    ds.Manufacturer = case_cfg["manufacturer"]
    ds.ManufacturerModelName = case_cfg["model"]
    ds.MagneticFieldStrength = case_cfg["magnetic_field"]
    ds.SliceThickness = case_cfg["slice_thickness"]
    ds.PixelSpacing = case_cfg["pixel_spacing"]
    ds.RepetitionTime = case_cfg["tr"]
    ds.EchoTime = case_cfg["te"]
    ds.WindowCenter = 128
    ds.WindowWidth = 256
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    # Image attributes
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = h
    ds.Columns = w
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = gray_img.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(str(out_file))
    print(f"Created DICOM: {out_file} ({w}x{h})")


def main():
    out_dir = Path("/home/sidd/Brain-Tumor-Detection/sample_dicoms")
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in DEMO_CASES:
        convert_jpg_to_dicom(case, out_dir)

    print("All clinical demo DICOM files created successfully in:", out_dir)


if __name__ == "__main__":
    main()
