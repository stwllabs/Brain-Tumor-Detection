# Brain Tumor MRI Classifier

A deep-learning prototype for classifying brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary.

> This is an academic prototype, not a medical diagnostic tool. Do not use its predictions to make clinical decisions.

## Run Locally

Install the dependencies and start the Streamlit app from the project root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The trained model artifacts are stored in `models/`. The MRI dataset in `data/` is only needed for training and is not required to run inference.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub, including the contents of `models/`. The `.gitignore` rules exclude the dataset but allow the model artifacts required by the app.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and select **Create app**.
3. Choose this repository and branch, then set the app entry point to `app/app.py`.
4. Select **Deploy**. Streamlit Cloud installs the dependencies from `requirements.txt` and builds the app.

The model artifacts are approximately 43 MB in total. If the GitHub repository has not received the latest changes yet, push them before creating the app. Do not commit the MRI dataset.

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://doi.org/10.21227/1jny-g144), mirrored on Kaggle as `masoudnickparvar/brain-tumor-mri-dataset`. It contains 7,023 images across four classes, split into `Training/` and `Testing/` directories.
