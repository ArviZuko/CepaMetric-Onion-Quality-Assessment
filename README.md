# CepaMetric — Onion AI Inspection System

CepaMetric is a conveyor-oriented onion quality inspection project developed by **Team CepaMetric** for Smart India Hackathon 2026 (problem statement **SIH26031**, Smart Automation). It aims to make onion grading more consistent by combining RGB-D measurement, object tracking, image classification, an operator dashboard, and sorting control. The supplied project includes Windows and Raspberry Pi application sources, an onion image dataset, five model-weight files, a Windows installer, a report, a presentation, and a demonstration video. The [project report](Documentation/Onion_AI_Inspection_System_Project_Report.docx) identifies the team as Team ID 167398 at Sanjivani University, Kopargaon.

## What the system is designed to do

Manual inspection can vary between operators and procurement centers. CepaMetric's proposed workflow separates **physical measurement** from **health classification**:

```text
Onion on conveyor
    → RGB + depth frame from an Intel RealSense D455f
    → contour/depth-based onion detection and persistent tracking ID
    → diameter estimate in millimetres and a size grade
    → cropped RGB image → CNN health prediction
    → optional Grad-CAM explanation and reflectance comparison
    → dashboard record and sorting decision
```

The report describes four health categories: **Moldy, Normal, Rotten, and Sprouted**. Its intended size bins are **below 40 mm**, **40–50 mm**, **50–60 mm**, and **above 60 mm**, with a reject route for diseased onions. The software source contains code for the dashboard, model inference, tracking, measurement, reflectance views, and related controls. The full camera-to-actuator workflow requires the original hardware and deployment assets; it was not independently validated when this repository was published.

## Why this approach

- **Hybrid inspection:** Classical computer vision and depth data locate, track, and measure onions. CNNs evaluate only the cropped onion image, reducing full-frame model work.
- **Physical size rather than pixel count:** The design uses camera geometry and depth to estimate diameter in millimetres. Calibration and camera placement remain essential.
- **Multiple classifiers:** ConvNeXt Tiny, EfficientNetV2-B0, DenseNet121, ResNet, and Xception provide four-class predictions. The source also offers probability averaging as an ensemble mode and references a stacking meta-classifier.
- **Reviewable decisions:** The application includes Grad-CAM generation and capture history so operators can inspect model outputs. The report additionally proposes a reflectance check against a reference set of normal onions.

These are design features, not claims that every branch is ready to run from a fresh clone. The [report](Documentation/Onion_AI_Inspection_System_Project_Report.docx) describes a broader integrated prototype than the portable files supplied here.

## Repository structure

```text
cepametric/
├── Dataset/
│   ├── train/                 labeled onion images
│   ├── valid/                 validation images
│   ├── test/                  test images
│   └── labels.csv             image and class metadata
├── SRC/
│   ├── Models/
│   │   ├── convnexttiny/      checkpoint in Release
│   │   ├── densenet121/      checkpoint in Git
│   │   ├── effnetv2/         checkpoint in Git
│   │   ├── Exception/        Xception checkpoint in Release
│   │   └── Resnet/           ResNet checkpoint in Release
│   ├── Raspi/                Raspberry Pi application and setup scripts
│   └── Windows/
│       ├── Src/              Windows PyQt application source
│       ├── Application/      Inno Setup project
│       └── Installer/        installer download guide
├── Documentation/           report, slides, and video download guide
└── Screenshots/              supplied interface and prototype images
```

The `Exception/` folder is the supplied folder name for the **Xception** model and is preserved as received.

### Dataset

The repository includes **2,539 image files** in four class folders, plus `Dataset/labels.csv`:

| Split | Molded | Normal | Rotten | Sprouted | Total images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 264 | 708 | 151 | 653 | 1,776 |
| Validation | 75 | 203 | 43 | 188 | 509 |
| Test | 38 | 102 | 21 | 93 | 254 |

The image folder uses **`molded`** while the application displays **Moldy**. `labels.csv` records source and split metadata. The report's statement that the raw dataset was not copied into its referenced repository is outdated for **this** repository: the above images are present here. Dataset provenance and reuse rights should be checked before redistribution.

### Model weights

| Model family | Supplied location | Delivery |
| --- | --- | --- |
| ConvNeXt Tiny | `SRC/Models/convnexttiny/` | [Release download guide](SRC/Models/convnexttiny/README.md) |
| EfficientNetV2-B0 | `SRC/Models/effnetv2/` | [Weight in Git](SRC/Models/effnetv2/effnetv2b0model.weights.h5) |
| DenseNet121 | `SRC/Models/densenet121/` | [Weight in Git](SRC/Models/densenet121/onion_densenet121_best_macro_f1.weights.h5) |
| ResNet | `SRC/Models/Resnet/` | [Release download guide](SRC/Models/Resnet/README.md) |
| Xception | `SRC/Models/Exception/` | [Release download guide](SRC/Models/Exception/README.md) |

The report calls the ResNet architecture **ResNet50V2**. The application code defines model builders and loads weights, but this repository does **not** contain the `onion_stacking_meta_classifier_best.keras` file referenced by the stacking option. Ensemble averaging has no separate checkpoint.

## Restoring large files

Five files exceeded GitHub's per-file Git limit. Download them from the [v1.0 large-files Release](https://github.com/ArviZuko/cepametric/releases/tag/v1.0-large-files). Each independent ZIP stores the original path and filename; **extract it at the repository root**.

| File | Folder guide |
| --- | --- |
| Windows installer | [`SRC/Windows/Installer/README.md`](SRC/Windows/Installer/README.md) |
| ConvNeXt Tiny weights | [`SRC/Models/convnexttiny/README.md`](SRC/Models/convnexttiny/README.md) |
| ResNet weights | [`SRC/Models/Resnet/README.md`](SRC/Models/Resnet/README.md) |
| Xception weights | [`SRC/Models/Exception/README.md`](SRC/Models/Exception/README.md) |
| Demonstration video | [`Documentation/README.md`](Documentation/README.md) |

## Application entry points and setup

- **Windows:** [`SRC/Windows/Src/app.py`](SRC/Windows/Src/app.py) is the Python source. An [Inno Setup definition](SRC/Windows/Application/OnionAIInspection.iss) and the [supplied installer](SRC/Windows/Installer/README.md) are also available. The code imports TensorFlow/Keras, PyQt5, OpenCV, NumPy, Matplotlib, and `pyrealsense2`; a Windows dependency lockfile is not supplied.
- **Raspberry Pi:** [`SRC/Raspi/final.py`](SRC/Raspi/final.py) is the application source. Start with [`setup.sh`](SRC/Raspi/setup.sh), [`requirements-pi.txt`](SRC/Raspi/requirements-pi.txt), and [`run.sh`](SRC/Raspi/run.sh). The supplied setup script targets a **64-bit Raspberry Pi 5** and installs system packages as well as Python dependencies. Review the script before running it on your device.

**Fresh-clone limitation:** The Windows source looks for model weights under a root-level `Training_DL_Models/Model_*/weights/` tree, whereas the supplied files are organized under `SRC/Models/`. The Raspberry Pi source also expects a separate `Weight Files/` directory beside `final.py`. Those paths must be reconciled for source execution. Both application sources reference a stacking checkpoint and external reflectance/reference assets that are absent from the supplied folder. Restoring the five Release ZIPs recreates the supplied folder, but does not create those missing assets. The installer was uploaded as provided and has not been independently tested on a clean machine.

## Reported results and current limits

The [project report](Documentation/Onion_AI_Inspection_System_Project_Report.docx) reports a **1.6 mm mean absolute diameter error across five onions** and **94% size-grade accuracy** for its prototype. It does not provide a reproducible classifier accuracy, Macro-F1, balanced accuracy, or confusion matrix for the files in this repository. Treat those numbers as report claims, not independently reproduced benchmarks.

RealSense sizing depends on calibration and scene geometry; model results depend on the dataset and compatible checkpoints. Before deployment, validate image provenance, the class mapping, the chosen camera and lighting, model paths, and sorting hardware. The report itself lists broader classifier evaluation and clean-machine packaging as remaining work.

## Documentation and attribution

- [Onion AI Inspection System project report](Documentation/Onion_AI_Inspection_System_Project_Report.docx)
- [CepaMetric project presentation](Documentation/CepaMetric_GitHub_Project_15_Slides.pptx)
- [Demonstration video download](Documentation/README.md)
- [Project screenshots](Screenshots/)

No project-level license was included in the supplied folder. Verify rights to the source, dataset, weights, report, and installer before reuse or redistribution.
