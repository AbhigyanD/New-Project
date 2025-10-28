# Credit Risk Analysis using Machine Learning

This project implements a machine learning pipeline for credit risk analysis using various ensemble models. The system analyzes financial text data to predict potential fraud cases.

## Features

- **Data Loading**: Fetches financial fraud dataset from Hugging Face
- **Text Processing**: Uses TF-IDF for text vectorization
- **Class Imbalance Handling**: Implements SMOTE for handling imbalanced classes
- **Multiple Models**: Includes Random Forest, XGBoost, LightGBM, and CatBoost
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Visualization**: Generates feature importance plots and ROC curves

## Results

| Model        | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------|----------|-----------|--------|----------|---------|
| Random Forest| 0.9706   | 1.0       | 0.9412 | 0.9697   | 1.0     |
| XGBoost      | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |
| LightGBM     | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |
| CatBoost     | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/credit-risk-analysis.git](https://github.com/yourusername/credit-risk-analysis.git)
   cd credit-risk-analysis


## Usage

```bash
python credit_risk_analysis.py
```

## License
MIT License

## Author
Abhigyan D

## Contact
ed.dey@mail.utoronto.ca

## Acknowledgments
- Hugging Face
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Imbalanced-learn

## References
- [Hugging Face](https://huggingface.co/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [CatBoost](https://catboost.ai/)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)

## Citation
```bibtex
@misc{credit-risk-analysis,
  author = {Abhigyan D},
  title = {Credit Risk Analysis using Machine Learning},
  howpublished = {\url{https://github.com/AbhigyanD/Credit-Risk-Analysis}},
  year = {2025}
}
```

## Contributing
Please feel free to contribute to this project by submitting pull requests or issues.

