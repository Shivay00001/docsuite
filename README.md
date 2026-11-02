# DocSuite - Enterprise Document AI Platform

DocSuite is a high-performance, production-grade Document AI platform designed for advanced OCR, layout analysis, and semantic data extraction. It features a robust SaaS-ready architecture with built-in licensing, usage tracking, and a modular core.

## 🚀 Features

- **Advanced OCR Pipeline**: Modular architecture with support for multiple backends (CRAFT, CRNN, etc.).
- **Smart Preprocessing**: Built-in denoising, deskewing, and image enhancement.
- **SaaS Architecture**:
  - **License Management**: Tiered licensing (Free/Pro/Enterprise) with offline validation support.
  - **Usage Tracking**: Local and sync-ready tracking of processed pages.
  - **JWT Authentication**: Secure API access with token-based auth.
- **Multi-Format Export**: Export results to TXT, JSON, CSV, DOCX, and Searchable PDF.
- **REST API**: Production-ready FastAPI implementation.
- **Standalone Executable**: Built with PyInstaller for one-click deployment.

## 📁 Project Structure

```text
document_ai/
├── core/               # OCR and Processing Core
│   ├── ocr/            # Engines, Detectors, Recognizers
│   └── licensing/      # License Manager & Usage Tracking
├── document/           # Document Logic
│   ├── loader/         # PDF and Image loading
│   └── exporter/       # Multi-format exporters
├── api/                # Interfaces
│   ├── rest/           # FastAPI Server
│   ├── auth/           # Security & Models
│   └── cli/            # CLI Command Logic
├── sdk.py              # Simple SDK interface
└── main.py             # CLI Entry Point
```

## 🛠️ Installation

### From Source

```bash
pip install -r requirements.txt
python -m document_ai.main --help
```

### Run API Server

```bash
python -m document_ai.main serve --port 8000
```

## 📦 Building the Executable

Run the included build script to generate a standalone Windows EXE:

```powershell
.\build.bat
```

Target: `dist/docsuite/docsuite.exe`

## ⚖️ License

**DocSuite Custom License**
Copyright (c) 2026 Shivay Singh

- **Personal Use**: Free of charge for personal, non-commercial use.
- **Commercial Use**: ANY use for direct or indirect financial gain requires a commercial license.
- Please refer to the `LICENSE` file for full terms.

---
Developed by **Shivay Singh**

## Built by VisionQuantech

Part of the [VisionQuantech](https://visionquantech.com/) AI automation studio. See [AI voice agents](https://visionquantech.com/ai-voice-agents) and the guide to [AI phone answering for real estate agents](https://visionquantech.com/blog/ai-phone-answering-service-for-real-estate-agents).
