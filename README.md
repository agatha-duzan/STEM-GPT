# STEM-GPT: Your Future AI Teaching Assistant?

**Author**: Agatha Duzan

---

## Overview

STEM-GPT is an AI assistant designed to answer multiple-choice questions (MCQ) in STEM subjects. This project aims to enhance the learning experience for EPFL students by providing accurate answers and detailed explanations.

**Key Highlights:**
- **Base Model**: Built upon GPT-Neo 125M, a lightweight yet powerful language model.
- **Techniques Used**:
  - Direct Preference Optimization (DPO) for aligning the model with human preferences.
  - Supervised Fine-Tuning (SFT) for improving multiple-choice question answering.
- **Final Model**: Open-source and optimized for running on personal devices, ensuring accessibility and privacy.

For detailed methodology and results, check the [project report](docs/STEM-GPT_Report.pdf).

---

## Motivation

With the growing capabilities of AI in education, STEM-GPT addresses challenges such as:
- Lack of accessible open-source AI tools for education.
- Data privacy concerns when using proprietary models like GPT-4.
- High computational cost of large-scale models.

STEM-GPT offers an open, lightweight alternative tailored for STEM education.

---

## Usage

1. **Setup**:
   - Clone the repository.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Training**:
   - Run DPO training: `python train_dpo.py`.
   - Fine-tune for MCQA: `python train_sft.py`.

3. **Evaluation**:
   - Evaluate MCQA accuracy: `python evaluate_mcqa.py`.

4. **Interactive Demo**:
   - Launch the interactive demo: `python demo.py`.

---




## Pipeline

The project workflow is summarized in the image below:

![Pipeline](pipeline.png)

---

## Results

- **Accuracy Highlights**:
  - Overall MCQA accuracy: **39%**.
  - Top-performing subject: Biology (**72% accuracy**).
- **Best Method**: Supervised Fine-Tuning (SFT) provided the highest accuracy for MCQ tasks.

Further evaluation details can be found in the [project report](docs/STEM-GPT_Report.pdf).

---

## Ethical Considerations

- Focused on accessibility and inclusivity, with plans for multilingual support.
- Transparent about limitations, including performance variability across subjects and the risk of bias.

---

## Usage

1. **Setup**:
   - Clone the repository.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Run Models**:
   - Fine-tune: `python train_sft.py`.
   - Evaluate: `python evaluate_mcqa.py`.

3. **Interactive Demo**:
   - Launch the assistant: `python demo.py`.

---

## Learn More

For additional information on datasets, evaluation metrics, and future work, refer to the [project report](docs/STEM-GPT_Report.pdf).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

