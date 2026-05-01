# Visual Question Answering (VQA)

This repository is a Visual Question Answering demo: upload an image, ask a natural-language question, and the model returns the top answers with confidence scores.

It includes:
- `vqa_transformer.pth` (trained model checkpoint)
- `word2idx.pkl` and `ans2idx.pkl` (vocabulary mappings used at training time)
- A Gradio web app (`app.py`) for interactive inference

Note: the checkpoint file is large, so this repo uses Git LFS.

## What is VQA?

Visual Question Answering (VQA) is a multi-modal task that combines computer vision and NLP. The model takes:
- an image, and
- a text question (e.g. “How many people are there?”)

and predicts an answer from a fixed answer vocabulary (classification). In this project, the image is encoded with a Transformer-style vision backbone and the question is encoded with a text Transformer; their representations are fused and used to predict the most likely answer.

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Git LFS (required to download `vqa_transformer.pth`)

Install Git LFS:
- macOS (Homebrew): `brew install git-lfs`
- Windows: install from https://git-lfs.com/

After installing: `git lfs install`

## Setup (macOS)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python test_setup.py

python app.py
```

Open: http://127.0.0.1:7860

## Setup (Windows)

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python test_setup.py

python app.py
```

Or double-click: `start_frontend.bat`

Open: http://127.0.0.1:7860

## Notes

- First run may download pretrained Vision Transformer weights via `torchvision`.
- If you want CUDA acceleration, install PyTorch using the official selector for your CUDA version (https://pytorch.org/get-started/locally/) and then install the remaining packages from `requirements.txt`.

## Optional: Override file locations

By default the app loads these files from the repo root:
- `vqa_transformer.pth`
- `word2idx.pkl`
- `ans2idx.pkl`

You can override paths with environment variables:
- `VQA_MODEL_PATH`
- `VQA_WORD2IDX_PATH`
- `VQA_ANS2IDX_PATH`

## 🔬 Model Performance

Expected performance on VQA v2.0 validation set:

- **Overall Accuracy**: ~62-65%
- **Yes/No Questions**: ~78-82%
- **Number Questions**: ~42-46%
- **Other Questions**: ~55-58%

Performance can be improved with:
- Longer training (50+ epochs)
- Larger batch sizes (if GPU memory allows)
- Data augmentation
- Answer vocabulary expansion
- Ensemble methods

## 💻 Inference API

### Python API

```python
from inference import VQAInference
from PIL import Image

# Initialize model
vqa = VQAInference(
    model_path='vqa_transformer.pth',
    word2idx_path='data/processed/word2idx.pkl',
    ans2idx_path='data/processed/ans2idx.pkl'
)

# Load image
image = Image.open('path/to/image.jpg')

# Ask a question
question = "What color is the car?"
predictions = vqa.predict(image, question, top_k=5)

# Display results
for answer, confidence in predictions:
    print(f"{answer}: {confidence:.2%}")
```

### Command Line

```bash
python inference.py --image path/to/image.jpg --question "What is this?"
```

## 🛠️ Configuration

Edit `config.yaml` to customize:

**Data Settings:**
```yaml
data:
  max_answers: 3000        # Answer vocabulary size
  max_question_len: 20     # Maximum question length
```

**Model Settings:**
```yaml
model:
  embed_dim: 256           # Embedding dimension
  hidden_dim: 512          # Hidden layer dimension
  num_heads: 8             # Attention heads
  n_layers: 2              # Transformer layers
```

**Training Settings:**
```yaml
training:
  batch_size: 128          # Batch size
  lr: 0.0005              # Learning rate
  epochs: 35               # Number of epochs
  optimizer: adamw         # Optimizer type
```

## 📦 Requirements

Main dependencies:
- `torch >= 2.0.0` - PyTorch deep learning framework
- `torchvision >= 0.15.0` - Vision models and transforms
- `gradio >= 3.0.0` - Web interface
- `Pillow >= 9.0.0` - Image processing
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `pyyaml` - Configuration parsing
- `tqdm` - Progress bars
- `scikit-learn` - Evaluation metrics

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Use CPU mode (slower): edit inference to use `device='cpu'`

### Model Not Found
- Ensure `vqa_transformer.pth` exists in project root
- Check that preprocessing was completed successfully

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Verify Python version: `python --version` (3.8+ required)

### Slow Inference
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`

## 📚 References

- **VQA v2.0 Dataset**: [Balanced VQA](https://visualqa.org/)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **MS COCO**: [Microsoft COCO Dataset](https://cocodataset.org/)
- **Attention Mechanism**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is for educational and research purposes. Please cite the VQA v2.0 dataset and MS COCO dataset if you use this code in your research.

## 🤝 Contributing

Contributions are welcome! Potential improvements:
- Add support for VQA v2.0 test set evaluation
- Implement beam search for answer generation
- Add more sophisticated attention mechanisms
- Create REST API endpoint
- Add model interpretability visualizations
- Support for additional datasets

## 📧 Contact

For questions or issues, please open an issue in the project repository.

## 🎓 Acknowledgments

- VQA v2.0 dataset creators
- PyTorch and torchvision teams
- MS COCO dataset contributors
- Gradio framework developers

---

**Made with ❤️ for Computer Vision and NLP**
