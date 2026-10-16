# CepaMetric

CepaMetric is an onion inspection project with Windows and Raspberry Pi application sources, five image-classification model families, an image dataset, documentation, and screenshots. The application sources describe camera-based classification, physical sizing, object tracking, and visual explanations. The supplied folder does not include a measured deployment-performance report in this README; consult the project report for the research details.

## Repository structure

| Path | Contents |
| --- | --- |
| [`Dataset/`](Dataset/) | `train`, `valid`, and `test` images plus `labels.csv` |
| [`SRC/Windows/Src/app.py`](SRC/Windows/Src/app.py) | Windows application source |
| [`SRC/Raspi/final.py`](SRC/Raspi/final.py) | Raspberry Pi application source |
| [`SRC/Raspi/`](SRC/Raspi/) | Raspberry Pi setup and launcher scripts and dependency list |
| [`SRC/Models/`](SRC/Models/) | ConvNeXt Tiny, DenseNet121, EfficientNetV2, ResNet, and Xception model folders |
| [`SRC/Windows/Installer/`](SRC/Windows/Installer/) | Windows installer download instructions |
| [`Documentation/`](Documentation/) | Project report, presentation, and demonstration video link |
| [`Screenshots/`](Screenshots/) | Supplied project screenshots |

## Download the large files

GitHub's per-file Git limit prevents five supplied files from being committed directly. They are preserved as independent ZIP files in the [v1.0 large-files Release](https://github.com/ArviZuko/cepametric/releases/tag/v1.0-large-files). Each ZIP contains the original path, so extract it at the **repository root** to restore the folder structure.

| Original location | Download instructions |
| --- | --- |
| Windows installer | [`SRC/Windows/Installer/README.md`](SRC/Windows/Installer/README.md) |
| ConvNeXt Tiny weights | [`SRC/Models/convnexttiny/README.md`](SRC/Models/convnexttiny/README.md) |
| ResNet weights | [`SRC/Models/Resnet/README.md`](SRC/Models/Resnet/README.md) |
| Xception weights | [`SRC/Models/Exception/README.md`](SRC/Models/Exception/README.md) |
| Demonstration video | [`Documentation/README.md`](Documentation/README.md) |

The DenseNet121 and EfficientNetV2 weights are below GitHub's per-file limit and are included directly in the repository. The dataset and other project files are also in Git.

## Getting started

For the Raspberry Pi build, begin with [`SRC/Raspi/setup.sh`](SRC/Raspi/setup.sh) and [`SRC/Raspi/requirements-pi.txt`](SRC/Raspi/requirements-pi.txt). That dependency file explicitly targets Raspberry Pi 5 with a 64-bit OS and Python 3.13; it is not a Windows dependency list. For the Windows build, the entry point is [`SRC/Windows/Src/app.py`](SRC/Windows/Src/app.py), and the supplied installer is available through the Release link above. The camera and operating-system dependencies need to match the target machine.

No project-level license was supplied with the source folder. Verify rights to the images, trained weights, and installer before redistribution or reuse.
