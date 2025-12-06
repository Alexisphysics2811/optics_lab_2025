# Optics Laboratory 2025

This repository contains Python code and analysis scripts from undergraduate physics optics laboratory experiments.

## Repository Structure

```
optics_lab_2025/
├── experiments/        # Individual lab experiment scripts
├── data_analysis/      # Reusable data analysis tools
├── utils/             # Utility functions and helpers
├── docs/              # Documentation and reports
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Getting Started

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Alexisphysics2811/optics_lab_2025.git
cd optics_lab_2025
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

Navigate to the experiments directory and run individual scripts:
```bash
cd experiments
python exp_01_example.py
```

## Dependencies

This project uses common scientific Python libraries:
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing and optimization
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation and analysis
- **scikit-image**: Image processing
- **lmfit**: Curve fitting

See `requirements.txt` for the complete list.

## Adding New Experiments

To add a new experiment:

1. Create a new script in the `experiments/` directory
2. Use descriptive naming (e.g., `exp_02_interference.py`)
3. Include docstrings and comments
4. Import utilities from `utils/` and `data_analysis/` as needed

## Organization Tips

- Store raw data in a local `data/` directory (not tracked by git)
- Save important plots and figures with descriptive names
- Document your experiments in the `docs/` directory
- Reuse common functions by adding them to `utils/` or `data_analysis/`

## Notes

This repository was created to organize and preserve Python code from PyCharm used during the optics laboratory course.

## License

This is a personal educational project. Code is provided as-is for reference and learning purposes.