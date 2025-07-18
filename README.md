# Threat Detection System

A comprehensive real-time threat detection system that combines multiple AI models to detect various security threats including weapons, fire/smoke, violence, and mask compliance.

## 🚀 Features

### Multi-Model Detection
- **Weapon Detection**: YOLO-based model trained to detect various types of weapons
- **Fire & Smoke Detection**: Real-time detection of fire and smoke hazards
- **Violence Detection**: LSTM-based model for detecting violent activities in video streams
- **Mask Detection**: Compliance monitoring for mask usage
- **Person Detection**: General person detection capabilities

### Deployment Options
- **Real-time Webcam Detection**: Direct camera feed analysis
- **Video File Processing**: Analysis of pre-recorded video files
- **Web-based Interface**: Flask web application for remote monitoring
- **Combined Detection**: Multi-model simultaneous detection

## 📁 Project Structure

```
Threat-Detection/
├── 📄 README.md                           # Main project documentation
├── 📄 requirements.txt                    # Python dependencies
├── 📄 PROJECT_STRUCTURE.md               # Project organization guide
│
├── 📁 src/                               # Source code directory
│   ├── 📁 core/                         # Core detection modules
│   │   ├── 📄 detect.py                 # Basic fire/smoke detection
│   │   ├── 📄 all.py                    # Multi-model detection
│   │   ├── 📄 vio.py                    # Violence detection
│   │   ├── 📄 videodetect.py            # Video detection utilities
│   │   └── 📄 detecting.py              # General detection utilities
│   │
│   ├── 📁 models/                       # AI model implementations
│   │   ├── 📄 lstm.py                   # LSTM-based video classification
│   │   └── 📄 newlstm.py                # Enhanced LSTM model
│   │
│   ├── 📁 web/                          # Web application modules
│   │   ├── 📄 live_feed.py              # Basic web interface
│   │   ├── 📄 improved_live_feed.py     # Enhanced web interface
│   │   └── 📄 vio_live_feed.py          # Violence detection web interface
│   │
│   └── 📁 utils/                        # Utility modules
│       ├── 📄 config.py                 # Configuration management
│       └── 📄 logger.py                 # Logging utilities
│
├── 📁 data/                             # Data and model storage
│   ├── 📁 models/                       # Trained model files
│   ├── 📁 samples/                      # Sample data for testing
│   ├── 📁 detect/                       # YOLO training outputs
│   └── 📁 yolo-Weights/                 # YOLO model weights
│
├── 📁 assets/                           # Media assets
│   ├── 📁 images/                       # Image files
│   └── 📁 videos/                       # Video files
│
├── 📁 scripts/                          # Utility scripts
│   ├── 📄 run_detection.py              # Main detection runner script
│   └── 📄 setup.py                      # Environment setup script
│
├── 📁 examples/                         # Example implementations
│   ├── 📄 basic_detection_example.py    # Basic detection example
│   └── 📄 web_interface_example.py      # Web interface example
│
├── 📁 notebooks/                        # Jupyter notebooks
│   ├── 📄 image_class.ipynb             # Image classification notebook
│   └── 📄 lstm.ipynb                    # LSTM model notebook
│
├── 📁 docs/                             # Documentation
│   ├── 📄 API_REFERENCE.md              # API documentation
│   └── 📄 DEPLOYMENT.md                 # Deployment guide
│
├── 📁 tests/                            # Test files (to be added)
├── 📁 logs/                             # Log files (created at runtime)
├── 📁 outputs/                          # Output files (created at runtime)
└── 📁 .venv/                            # Virtual environment
```

## 🔧 Installation

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd Threat-Detection

# Run automated setup
python scripts/setup.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs outputs

# Verify model files exist
python -c "from src.utils.config import Config; print('Models found:', len(Config.get_model_paths()))"
```

## 🚀 Usage

### Main Detection Runner
The easiest way to run the system is using the main runner script:

```bash
# Basic fire/smoke detection
python scripts/run_detection.py basic

# Violence detection
python scripts/run_detection.py violence

# LSTM-based video classification
python scripts/run_detection.py lstm

# Multi-model detection (all models)
python scripts/run_detection.py all

# Web-based interface
python scripts/run_detection.py web
```

### Individual Module Usage

#### 1. Basic Object Detection
```bash
python src/core/detect.py
```
- Detects fire and smoke using webcam feed
- Press 'q' to quit

#### 2. Violence Detection
```bash
python src/core/vio.py
```
- Analyzes video files for violent activities
- Uses LSTM model for sequence-based detection

#### 3. LSTM-based Video Classification
```bash
python src/models/lstm.py
```
- Real-time video classification using LSTM
- Requires pre-trained LSTM model

#### 4. Combined Multi-Model Detection
```bash
python src/core/all.py
```
- Runs all detection models simultaneously
- Provides comprehensive threat analysis

#### 5. Web-based Interface
```bash
cd src/web
python live_feed.py
```
- Access at `http://localhost:5000/live_feed`
- Real-time web-based monitoring

### Example Implementations

#### Basic Detection Example
```bash
python examples/basic_detection_example.py
```
- Demonstrates basic fire/smoke detection
- Shows how to integrate with your own code

#### Web Interface Example
```bash
python examples/web_interface_example.py
```
- Runs the web-based monitoring interface
- Accessible from any device on the network

## 📁 Directory Details

### `src/` - Source Code
**Core detection modules and AI implementations**

#### `src/core/` - Core Detection Modules
- **`detect.py`**: Basic fire/smoke detection using YOLO
- **`all.py`**: Multi-model detection combining all systems
- **`vio.py`**: Violence detection using LSTM
- **`videodetect.py`**: Video processing utilities
- **`detecting.py`**: General detection utilities

#### `src/models/` - AI Model Implementations
- **`lstm.py`**: LSTM-based video classification
- **`newlstm.py`**: Enhanced LSTM model with improvements

#### `src/web/` - Web Applications
- **`live_feed.py`**: Basic Flask web interface
- **`improved_live_feed.py`**: Enhanced interface with multiple models
- **`vio_live_feed.py`**: Violence detection web interface

#### `src/utils/` - Utility Modules
- **`config.py`**: Centralized configuration management
- **`logger.py`**: Professional logging system

### `data/` - Data and Models
**All trained models and data storage**

#### `data/models/` - Trained Model Files
- **`fire.onnx`**: Fire detection ONNX model
- **`weapon1.onnx`**: Weapon detection ONNX model 1
- **`weapon2.onnx`**: Weapon detection ONNX model 2
- **`mask.onnx`**: Mask detection ONNX model
- **`video_classifier_model_lstm.h5`**: LSTM video classifier
- **`maskdetection.keras`**: Mask detection Keras model

#### `data/detect/` - YOLO Training Outputs
- **`train/`**: General detection training results
- **`fire_smoke_train/`**: Fire/smoke detection training
- **`weapon2_detection_train/`**: Weapon detection training 2
- **`weapondetction1_train/`**: Weapon detection training 1
- **`lstm/`**: LSTM model files
- **`val/`**: Validation results
- **`modelnew.h5`**: Violence detection model

#### `data/yolo-Weights/` - YOLO Weights
- **`yolov8n.pt`**: YOLO v8 nano weights

#### `data/samples/` - Sample Data
- Directory for test data and samples

### `assets/` - Media Assets
**Images and videos for testing and demonstration**

#### `assets/images/` - Image Files
- Sample images for testing detection systems
- Various formats: JPEG, JPG

#### `assets/videos/` - Video Files
- Sample videos for testing violence detection
- Various scenarios: fights, explosions, weapons

### `scripts/` - Utility Scripts
**Main entry points and automation tools**

#### `scripts/run_detection.py` - Main Detection Runner
- Unified interface for all detection modes
- Easy command-line access to all features
- Supports: basic, violence, lstm, all, web modes

#### `scripts/setup.py` - Environment Setup
- Automated dependency installation
- Directory creation
- Model verification
- Environment validation

### `examples/` - Example Implementations
**Ready-to-run examples and demonstrations**

#### `examples/basic_detection_example.py`
- Basic fire/smoke detection example
- Shows integration with custom code
- Educational implementation

#### `examples/web_interface_example.py`
- Web interface demonstration
- Network-accessible monitoring
- Production-ready example

### `notebooks/` - Jupyter Notebooks
**Interactive development and analysis**

#### `notebooks/image_class.ipynb`
- Image classification development
- Model training and evaluation
- Interactive data analysis

#### `notebooks/lstm.ipynb`
- LSTM model development
- Video classification experiments
- Model performance analysis

### `docs/` - Documentation
**Comprehensive project documentation**

#### `docs/API_REFERENCE.md`
- Complete API documentation
- Module usage examples
- Configuration options
- Error handling guide

#### `docs/DEPLOYMENT.md`
- Production deployment guide
- Cloud deployment instructions
- Docker containerization
- Performance optimization

### `tests/` - Test Files
**Unit tests and integration tests**
- Ready for test implementation
- Framework for quality assurance

### `logs/` - Log Files
**Runtime logging and monitoring**
- Created automatically during execution
- System performance logs
- Detection event logs

### `outputs/` - Output Files
**Generated outputs and results**
- Detection results
- Performance metrics
- Analysis outputs

## 🎯 Detection Capabilities

### Object Detection Models
- **YOLO Models**: Multiple trained YOLO models for different threat types
- **Confidence Thresholds**: Configurable detection confidence levels
- **Real-time Processing**: Live video stream analysis

### Violence Detection
- **LSTM Architecture**: Sequence-based violence detection
- **Feature Extraction**: InceptionV3 for frame feature extraction
- **Temporal Analysis**: Analyzes video sequences for violent patterns

### Web Interface Features
- **Live Streaming**: Real-time video feed in web browser
- **Multi-model Support**: Simultaneous detection from multiple models
- **Responsive Design**: Works on various screen sizes

## 🔧 Configuration

### Using the Configuration System
```python
from src.utils.config import Config

# Get all model paths
model_paths = Config.get_model_paths()

# Check if models exist
missing_models = Config.check_models_exist()

# Get class names for different models
class_names = Config.get_class_names()

# Access detection settings
confidence_threshold = Config.CONFIDENCE_THRESHOLD
image_size = Config.IMAGE_SIZE
```

### Model Paths
Update model paths in `src/utils/config.py` as needed:
```python
# YOLO Models
FIRE_SMOKE_MODEL = "data/detect/fire_smoke_train/weights/best.pt"
WEAPON_MODEL_1 = "data/detect/weapondetction1_train/weights/best.pt"
WEAPON_MODEL_2 = "data/detect/weapon2_detection_train/weights/best.pt"

# ONNX Models
FIRE_ONNX = "data/models/fire.onnx"
WEAPON1_ONNX = "data/models/weapon1.onnx"
WEAPON2_ONNX = "data/models/weapon2.onnx"
```

### Detection Parameters
- **Confidence Threshold**: Adjust detection sensitivity (default: 0.4)
- **Frame Size**: Modify input resolution as needed
- **Model Selection**: Enable/disable specific detection models

## 📊 Model Training

The project includes trained models for:
- **Weapon Detection**: Custom dataset training
- **Fire/Smoke Detection**: Hazard detection training
- **Violence Detection**: LSTM-based sequence training
- **General Object Detection**: YOLO-based training

Training outputs include:
- Model weights (`.pt` files)
- Training metrics and plots
- Confusion matrices
- Performance evaluation results

## 🛠️ Technical Requirements

- **Python**: 3.7+
- **OpenCV**: Computer vision operations
- **Ultralytics YOLO**: Object detection
- **TensorFlow/Keras**: Deep learning models
- **Flask**: Web application framework
- **NumPy**: Numerical computations

## 🔍 Troubleshooting

### Common Issues
1. **Camera Access**: Ensure camera permissions are granted
2. **Model Loading**: Verify model weight files are present
3. **Dependencies**: Install all required packages
4. **Memory**: Large models may require significant RAM

### Performance Optimization
- Reduce frame resolution for better performance
- Use GPU acceleration if available
- Adjust confidence thresholds based on requirements

### Using the Logger
```python
from src.utils.logger import logger

# Log system events
logger.info("System started")
logger.detection_event("weapon", 0.95, "camera_1")
logger.error("Detection failed")
```

## 📝 License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when deploying in production environments.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.

### Development Workflow
1. **Core functionality** → `src/core/`
2. **AI models** → `src/models/`
3. **Web interfaces** → `src/web/`
4. **Utilities** → `src/utils/`
5. **Documentation** → `docs/`

## 📞 Support

For questions and support, please open an issue in the repository or contact the development team.

### Quick Help
- **Setup Issues**: Run `python scripts/setup.py`
- **Model Issues**: Check `data/models/` directory
- **Web Interface**: Access `http://localhost:5000/live_feed`
- **Documentation**: See `docs/` directory
