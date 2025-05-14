# llama3.2-vision_Handwritten_Number_Recognition

Building on the previous approach, this version leverages Meta’s LLaMA 3.2 Vision model to achieve more accurate recognition of handwritten marks on exam answer sheets.

This enhanced system is especially effective in cases where handwriting is inconsistent or hard to interpret, making it highly suitable for large-scale digitization of exam data.

---

## How It Works

1. A clean reference image of a blank answer sheet is used to define Region of Interest (ROI) coordinates for mark entry areas.
2. The extracted regions are passed to the LLaMA 3.2 Vision model, which interprets and returns the handwritten number.
3. Multi-digit numbers are fully supported.
4. The predictions are saved in a structured CSV file for further analysis or record-keeping.

---

## Features

-  Powered by Meta’s LLaMA 3.2 Vision large multimodal model
-  High accuracy even with poor handwriting
-  Supports batch processing of multiple answer sheets
-  Results saved in an easy-to-use CSV format

---

## Project Structure
```
llama3.2-vision_Handwritten_Number_Recognition/
│
├── input_images/ # Folder containing scanned answer sheet images
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── reference_image.jpg # Blank answer sheet used to define ROIs
├── roi_values.txt # ROI coordinates manually defined
├── llama3.2-vision_number_recognition.py # Main script for running predictions
├── output.csv # Predicted results
├── requirements.txt # Required packages and dependencies
├── README.md # Project documentation
└── LICENSE # Project license (MIT)
```
---

## How to run

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   
2. Run the script:
   ```bash
   python llama3.2-vision_number_recognition.py

3. View predictions in output.csv.

---

## Prerequisites

 - Python 3.10+
 - GPU with minimum 4GB VRAM
 - Ollama setup to access LLaMA 3.2 Vision
 - OpenCV, pandas, and other standard libraries (available in requirements.txt)

---

## License

This project is licensed under the MIT License.
