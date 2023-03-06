# Pedestrian detection

Pedestrian detection module is based on the "Domain adaption on pedestrian detection with Faster R-CNN" paper written by Matteo Sirri for the "School in AI: Deep Learning, Vision and Language for Industry - second edition" final project work [@UNIMORE](https://www.unimore.it)

## Demo Links

|                                                                    Google Colab Demo                                                                    |                                                                       Huggingface Demo                                                                        |                                                    Report                                                     |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KQqmPANWiLqAJH0yZN1UV_FVqnzPrurw) | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sir3mat/SchoolInAiProjectWork) | [Report](https://docs.google.com/document/d/1U0yEuGx5wJ8xkZUpdMQS59XM9V7IidX-vzX9bOh6iEM/edit?usp=share_link) |

- Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio).

## Object Detection

An adaption of torchvision's detection reference code is done to train Faster R-CNN on a portion of the MOTSynth dataset.

- To train the model you can run (change params in the script):

```
./scripts/detector/train_detector.sh
```

- To fine-tuning the model you can run (change params in the script):

```
./scripts/detector/fine_tuning_detector.sh
```

- To evaluate the model you can run (change params in the script):

```
./scripts/detector/evaluate_detector.sh
```

- To make inference and show results you can run (change params in the script):

```
./scripts/detector/inference_detector.sh
```
