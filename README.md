# STEM-GPT: Your Future AI Teaching Assistant?

**Author**: Agatha Duzan  
**Email**: [agatha.duzan@epfl.ch](mailto:agatha.duzan@epfl.ch)  
**Repository**: [agatha-duzan](#)

---

## Overview

STEM-GPT is an AI assistant designed to answer multiple-choice questions (MCQ) in STEM subjects. By leveraging advanced language models, this project aims to enhance the learning experience for EPFL students by providing accurate answers and detailed explanations.

**Key Highlights:**
- **Base Model**: Built upon GPT-Neo 125M, a lightweight yet powerful language model.
- **Techniques Used**:
  - Direct Preference Optimization (DPO) for aligning the model with human preferences.
  - Supervised Fine-Tuning (SFT) for improving multiple-choice question answering.
- **Performance**:
  - Achieves an overall accuracy of **39%** on MCQ tasks.
  - Excels in biology (**72% accuracy**) but struggles in math and physics.
- **Final Model**: Open-source and optimized for running on personal devices, ensuring accessibility and privacy.

---

## Motivation

With the growing capabilities of AI in education, STEM-GPT addresses challenges such as:
- Lack of accessible open-source AI tools for education.
- Data privacy concerns when using proprietary models like GPT-4.
- High computational cost of large-scale models.

STEM-GPT offers an open, lightweight alternative tailored for STEM education.

---

## Methodology

1. **Model Training**:
   - **Base Model**: GPT-Neo 125M was selected for its balance between performance and computational efficiency.
   - **DPO Training**:
     - Used preference pairs collected from EPFL courses.
     - Improved model alignment with human preferences.
   - **SFT**:
     - Fine-tuned the model on STEM-related MCQ datasets to optimize accuracy.

2. **Datasets**:
   - EPFL STEM MCQ dataset (356 questions).
   - Open-source benchmarks:
     - **MMLU** (3,429 STEM-related questions).
     - **ARC Challenge** (2,590 advanced questions).

3. **Evaluation**:
   - Reward accuracy (measuring alignment with human preferences).
   - MCQA accuracy (proportion of correct answers).

---

## Results

- **Performance Metrics**:
  - Reward accuracy: **56%** (after DPO).
  - MCQA accuracy: **39%** (using SFT).

- **Subject-Specific Accuracy**:
  - Biology: **72%**
  - Chemistry: **55%**
  - Computer Science: **25%**
  - Math & Physics: **<20%**

- **Adaptation Methodologies**:
  - DPO showed marginal improvement but was less impactful than SFT for MCQA tasks.
  - SFT alone yielded the best results.

---

## Ethical Considerations

- **Inclusivity**:
  - Planned adaptation for other languages and signed languages.
  - Potential use of transfer learning for low-resource languages.

- **Limitations**:
  - Risk of propagating misinformation.
  - Performance disparity in non-English applications.

- **Mitigations**:
  - Transparency about limitations.
  - Fairness evaluations across diverse groups.

---

## Future Work

- Extend model capabilities to include multimodal inputs (e.g., interpreting graphs and diagrams).
- Develop a multilingual and diverse STEM dataset.
- Explore larger, more advanced open-source models.

---

## Repository Structure

- `src/`: Source code for training and evaluation.
- `data/`: Preprocessed datasets.
- `results/`: Model outputs and evaluation results.
- `docs/`: Documentation, including this `README.md` and the full project report ([`STEM-GPT_Report.pdf`](docs/STEM-GPT_Report.pdf)).

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

## Acknowledgments

This project was inspired by advancements in natural language processing and fine-tuning techniques. Special thanks to EPFL for providing datasets and resources, and to EleutherAI for developing GPT-Neo.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
