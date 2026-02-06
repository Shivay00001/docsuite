# User Interface Specification

## Overview

This document outlines the architecture for the **DocSuite** user interface, designed to work as a hybrid Desktop (Electron) and Web application. The UI communicates with the core Python backend via the REST API.

## Architecture

### Hybrid Approach

- **Frontend**: React + TypeScript (built via Vite).
- **Desktop Wrapper**: Electron (wraps the React app).
- **Backend**: Existing FastAPI Python server (`document_ai/api/rest/app.py`).

### Communication Flow

1. **Desktop**: Electron starts the Python server as a sidecar process (managed child process).
2. **Web**: Python server runs independently; React app connects via HTTP.
3. **Data**: JSON for metadata, Multipart for file uploads.

## Feature Specifications

### 1. Dashboard (`/`)

- **Status Widget**: Server health, License status (Free/Pro/Enterprise).
- **Usage Stats**: Documents processed today, remaining quota.
- **Quick Action**: Drag-and-drop info zone for immediate OCR.

### 2. Document Studio (`/studio`)

- **Upload Area**: Large drop zone supporting PDF, PNG, JPG, TIFF.
- **Processing State**: Progress bar with stage indicators (Preprocessing -> OCR -> Export).
- **Result Viewer**:
  - **Split View**: Original image (left) vs Extracted text (right).
  - **Overlay Mode**: Bounding boxes drawn over original image.
  - **Edit Mode**: Manually correct OCR errors.

### 3. Batch Processor (`/batch`)

- **Queue Manager**: List of files to process.
- **Bulk Settings**: Select export formats (PDF, JSON, CSV) for entire batch.
- **Background Mode**: minimize to tray while processing.

### 4. Settings (`/settings`)

- **Licensing**: Input license key, view active plan.
- **Engine**: Select OCR backend (Tesseract/EasyOCR/Custom), toggle GPU acceleration.
- **Output**: Default export paths and naming conventions.
- **Appearance**: Dark/Light mode toggle.

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | React 18 |
| **Build Tool** | Vite |
| **Styling** | TailwindCSS + Radix UI (for accessibility) |
| **State** | Zustand (lightweight global state) |
| **Desktop** | Electron 28 |
| **Icons** | Lucide React |

## Mockups (ASCII)

```
+-------------------------------------------------------+
|  DocSuite Pro                            [ - ] [ x ]  |
+-------------------+-----------------------------------+
| [ Dashboard     ] |  Warning: License Expiring Soon   |
| [ Studio        ] |                                   |
| [ Batch         ] |  +-----------------------------+  |
| [ Settings      ] |  |      DROP FILES HERE        |  |
|                   |  |      OR CLICK TO BROWSE     |  |
|                   |  +-----------------------------+  |
|                   |                                   |
|                   |  Recent Documents:                |
| [ User: Admin   ] |  - invoice_2023.pdf (Processed)   |
+-------------------+-----------------------------------+
```
