from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import shutil
import time
import requests
from PIL import Image
import numpy as np
import logging
import torch
from io import BytesIO
import json
from pathlib import Path
import mimetypes
import traceback  # ADD THIS IMPORT
import pandas as pd  # ADD THIS IMPORT
import subprocess  # ADD THIS IMPORT FOR RUNNING EXTERNAL COMMANDS
import socket      # ADD THIS IMPORT FOR PORT CHECKING

# Import calc_prop functions
from calc_prop import process_mask

# Try to import PharmaMCP (if available)
PHARMAMCP_AVAILABLE = False
try:
    from pharmadissolve_mcp import PharmaDissolveMCP
    PHARMAMCP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ PharmaMCP module imported successfully")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PharmaMCP module not available. Dissolution prediction will be disabled.")

# Fix for tifffile deprecation
try:
    import tifffile
    if not hasattr(tifffile, 'imsave'):
        tifffile.imsave = tifffile.imwrite
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cellpose imports with compatibility patches
try:
    from cellpose_omni import io, models
    logger.info("✅ cellpose_omni imported successfully")
    
    # Verify omnipose compatibility
    try:
        from omnipose.core import OMNI_MODELS
    except ImportError:
        logger.warning("⚠️ Patching omnipose for compatibility")
        import omnipose.core
        omnipose.core.OMNI_MODELS = ['bact', 'bact_omni', 'cyto', 'cyto2', 'cyto2_omni']
        
except ImportError:
    logger.critical("❌ Failed to import cellpose_omni")
    raise

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
DISSOLUTION_PREDICTIONS_DIR = "dissolution_predictions"  # ADD THIS LINE
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DISSOLUTION_PREDICTIONS_DIR, exist_ok=True)  # ADD THIS LINE

# Clear existing models
MODEL_DIR = os.path.expanduser("~/.cellpose/models")
if os.path.exists(MODEL_DIR):
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
        logger.info("✅ Cleared existing models")
    except Exception as e:
        logger.error(f"❌ Failed to clear models: {str(e)}")

# Load model with retries
model = None
MAX_RETRIES = 5

def verify_model_file(model_path):
    """Verify model file integrity"""
    if not os.path.exists(model_path):
        return False
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"Model size: {file_size:.2f} MB")
    return 20 < file_size < 30

for attempt in range(MAX_RETRIES):
    try:
        logger.info(f"Attempt {attempt+1}/{MAX_RETRIES}: Loading cyto2 model")
        model = models.CellposeModel(gpu=False, model_type="cyto2")
        if not model.pretrained_model or not verify_model_file(model.pretrained_model[0]):
            raise ValueError("Invalid model file")
        logger.info("✅ Model loaded successfully")
        break
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model_path = os.path.join(MODEL_DIR, "cyto2torch_0")
        if os.path.exists(model_path):
            os.remove(model_path)
        response = requests.get("https://www.cellpose.org/models/cyto2torch_0", stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(5)  # Allow filesystem sync

# ADD THIS FUNCTION
def get_pharma_mcp_instance():
    """Get PharmaMCP instance with proper configuration"""
    api_key = os.getenv("PHARMAMCP_API_KEY", "")
    model_name = os.getenv("PHARMAMCP_MODEL", "deepseek/deepseek-chat-v3.1:free")
    kb_path = os.getenv("PHARMAMCP_KB_PATH", "RAG_database.xlsx")
    persist_dir = os.getenv("PHARMAMCP_PERSIST_DIR", "faiss_index")
    
    # Ensure knowledge base exists
    if not os.path.exists(kb_path):
        logger.warning(f"⚠️ Knowledge base not found at {kb_path}. Creating minimal example.")
        try:
            # Create a minimal example knowledge base
            df = pd.DataFrame({
                "Drug Name": ["ExampleDrug"],
                "Particle Size (μm)": [10.0],
                "Solubility (mg/mL)": [5.0],
                "Time (min)": [0, 5, 10, 15, 30],
                "Dissolved (%)": [0, 25, 50, 75, 95]
            })
            df.to_excel(kb_path, index=False)
            logger.info(f"✅ Created minimal knowledge base at {kb_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create minimal knowledge base: {str(e)}")
    
    return PharmaDissolveMCP(
        api_key=api_key,
        model_name=model_name,
        kb_path=kb_path,
        persist_dir=persist_dir
    )

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

@app.post("/upload-and-segment")
async def upload_and_segment(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(500, "Segmentation service unavailable")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "Invalid image format")
        
        # Read and process image
        img_pil = Image.open(BytesIO(await file.read()))
        img_np = np.array(img_pil.convert('L'))
        logger.info(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        
        # Enhanced segmentation parameters
        masks, flows, styles = model.eval(
            img_np,
            channels=[0, 0],
            omni=True,
            invert=False,
            diameter=15,
            mask_threshold=0.0,
            flow_threshold=0.0,
        )
        
        # Create results directory
        out_id = str(uuid.uuid4())
        result_dir = os.path.join(RESULT_DIR, out_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save results
        io.save_masks(
            [img_np], [masks], [flows],
            [os.path.join(UPLOAD_DIR, f"{out_id}.png")],
            savedir=result_dir,
            save_ncolor=True,
            save_outlines=True,
            save_flows=True,
            tif=True  # Save Tiff files
        )
        
        # Save the raw mask as .npy for calc_prop.py (critical for compatibility)
        mask_path = os.path.join(result_dir, f"{out_id}_mask.npy")
        np.save(mask_path, masks)
        
        # Create dissolution results directory
        dissolution_dir = os.path.join(result_dir, "dissolution_prop_results")
        os.makedirs(dissolution_dir, exist_ok=True)
        
        # Automatically run particle analysis
        try:
            process_mask(
                mask_path=mask_path,
                output_dir=dissolution_dir,
                scale=1/60  # Default scale factor - adjust as needed
            )
            logger.info("✅ Particle analysis completed successfully")
        except Exception as e:
            logger.error(f"❌ Particle analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Initialize response data
        response_data = {
            "id": out_id,
            "status": "success",
            "logs": "Segmentation and analysis completed",
            "images": {
                "output": f"/result/{out_id}/{out_id}_cp_output.png",
                "masks_ncolor": f"/result/{out_id}/{out_id}_cp_ncolor_masks.png",
                "outlines": f"/result/{out_id}/{out_id}_outlines.png",
                "flows": f"/result/{out_id}/{out_id}_flows.tif",
                "masks_tif": f"/result/{out_id}/{out_id}_dP.tif",
                "mask_npy": f"/result/{out_id}/{out_id}_mask.npy"
            },
            "mask_count": int(masks.max()) if masks.max() > 0 else 0,
            "analysis_results": {
                "excel": f"/result/{out_id}/dissolution_prop_results/particle_analysis.xlsx",
                "json": f"/result/{out_id}/dissolution_prop_results/properties.json",
                "plots": f"/result/{out_id}/dissolution_prop_results/distributions.png"
            }
        }
        
        # Add PharmaMCP prediction if available
        if PHARMAMCP_AVAILABLE:
            try:
                # Load particle analysis results
                properties_path = os.path.join(dissolution_dir, "properties.json")
                if os.path.exists(properties_path):
                    with open(properties_path, 'r') as f:
                        props = json.load(f)
                    
                    # Get additional parameters from Excel if available
                    excel_path = os.path.join(dissolution_dir, "particle_analysis.xlsx")
                    aspect_ratio = None
                    if os.path.exists(excel_path):
                        df = pd.read_excel(excel_path, sheet_name="Particles")
                        if "Aspect Ratio" in df.columns:
                            aspect_ratio = df["Aspect Ratio"].mean()
                    
                    # Determine shape based on aspect ratio
                    shape = "irregular"
                    if aspect_ratio is not None:
                        if aspect_ratio < 1.5:
                            shape = "spherical"
                        elif aspect_ratio < 2.5:
                            shape = "ellipsoidal"
                    
                    # Get solubility from environment or default value
                    solubility = float(os.getenv("DEFAULT_SOLUBILITY", "10.0"))
                    
                    # Format query for PharmaMCP
                    query = (
                        f"Predict dissolution profile for Drug Dissolved (%) "
                        f"Particle Size: {props['d50']:.2f} μm "
                        f"Solubility: {solubility} mg/mL "
                        f"Shape: \"{shape}\""
                    )
                    
                    logger.info(f"💊 PharmaMCP query: {query}")
                    
                    # Run PharmaMCP prediction
                    pharma_mcp = get_pharma_mcp_instance()
                    prediction = pharma_mcp.run(
                        query=query,
                        n_candidates=3
                    )
                    
                    # Copy prediction files to our directory for serving
                    report_filename = f"{out_id}_dissolution_report.md"
                    profile_filename = f"{out_id}_dissolution_profile.json"
                    
                    shutil.copy(prediction['report_path'], os.path.join(DISSOLUTION_PREDICTIONS_DIR, report_filename))
                    shutil.copy(prediction['profile_path'], os.path.join(DISSOLUTION_PREDICTIONS_DIR, profile_filename))
                    
                    # Add prediction results to response
                    response_data["dissolution_prediction"] = {
                        "report": f"/prediction/{report_filename}",
                        "profile": f"/prediction/{profile_filename}",
                        "qc_metrics": prediction["qc_metrics"],
                        "d50": props['d50']  # Include the particle size used
                    }
                    
                    # CORRECTED LOGGING - Handle None values properly
                    qc_metrics = prediction["qc_metrics"]
                    f2_value = qc_metrics.get('f2')
                    t50_value = qc_metrics.get('T50')
                    t90_value = qc_metrics.get('T90')
                    
                    logger.info(f"✅ Dissolution prediction completed for {out_id}")
                    logger.info(f"   F2 similarity: {f2_value:.2f}" if f2_value is not None else "   F2 similarity: N/A")
                    logger.info(f"   T50: {t50_value:.2f} min" if t50_value is not None else "   T50: N/A min")
                    logger.info(f"   T90: {t90_value:.2f} min" if t90_value is not None else "   T90: N/A min")
                    
                    # ============== NEW: Dashboard Generation ==============
                    logger.info("🔄 Generating dashboard from PharmaMCP results...")
                    try:
                        # Generate the dashboard
                        subprocess.run([
                            "python", "dashboard_gallary.py",
                            "--excel", "RAG_database.xlsx",
                            "--log", "mcp_runs.jsonl",
                            "--out", "dashboards_basic"
                        ], check=True)
                        logger.info("✅ Dashboard generated successfully at dashboards_basic/")
                        
                        # Check if dashboard server is already running
                        if not is_port_in_use(8090):
                            logger.info("🚀 Starting dashboard server on port 8090...")
                            # Start the HTTP server in a background process
                            subprocess.Popen([
                                "python", "-m", "http.server", 
                                "-d", "dashboards_basic", "8090"
                            ])
                            logger.info("✅ Dashboard server started on http://localhost:8090")
                        else:
                            logger.info("ℹ️ Dashboard server is already running on port 8090")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"❌ Dashboard generation failed: {str(e)}")
                    except Exception as e:
                        logger.error(f"❌ Error during dashboard generation: {str(e)}")
                    # ============== END NEW SECTION ==============
                else:
                    logger.warning(f"⚠️ Properties JSON not found at {properties_path}")
            except Exception as e:
                logger.error(f"❌ PharmaMCP prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
        
        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# Add endpoint for dissolution predictions
@app.get("/prediction/{filename}")
async def get_prediction_file(filename: str):
    """Serve dissolution prediction files"""
    path = os.path.join(DISSOLUTION_PREDICTIONS_DIR, filename)
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    if filename.endswith(".md"):
        media_type = "text/markdown"
    elif filename.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(path, media_type=media_type)

# FIXED: Properly handle files in subdirectories
@app.get("/result/{out_id}/{filename:path}")
async def get_file(out_id: str, filename: str):
    """
    Serve files from RESULT_DIR/<out_id>/ safely.
    - Prevents path traversal by resolving absolute paths and ensuring the file is inside RESULT_DIR.
    - Handles files in subdirectories (like dissolution_prop_results)
    - Returns 404 if missing, 400 for bad requests.
    """
    # Build and resolve path
    result_dir = Path(RESULT_DIR).resolve()
    requested = (result_dir / out_id / filename).resolve()

    # Security check: requested path must remain inside RESULT_DIR
    if result_dir not in requested.parents:
        logger.error(f"Invalid path traversal attempt: {filename}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Existence check
    if not requested.is_file():
        logger.error(f"File not found: {requested}")
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on file extension
    media_type, _ = mimetypes.guess_type(requested.name)
    if media_type is None:
        # Provide common fallbacks
        ext = requested.suffix.lower()
        media_type = {
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".npy": "application/octet-stream",
        }.get(ext, "application/octet-stream")

    # For TIFFs and similar, force download
    headers = {}
    if requested.suffix.lower() in (".tif", ".tiff", ".xlsx"):
        headers["Content-Disposition"] = f'attachment; filename="{requested.name}"'

    return FileResponse(path=requested, media_type=media_type, headers=headers)

if __name__ == "__main__":
    import uvicorn
    # Start the main application
    logger.info("🚀 Starting main application on http://localhost:8000")
    
    # Check if dashboard server is already running
    if not is_port_in_use(8090):
        logger.info("💡 Dashboard server not detected, starting it now...")
        try:
            # Generate initial dashboard if it doesn't exist
            if not os.path.exists("dashboards_basic"):
                logger.info("🔄 Generating initial dashboard...")
                subprocess.run([
                    "python", "dashboard_gallary.py",
                    "--excel", "RAG_database.xlsx",
                    "--log", "mcp_runs.jsonl",
                    "--out", "dashboards_basic"
                ], check=True)
            
            # Start the dashboard server
            logger.info("🚀 Starting dashboard server on port 8090...")
            subprocess.Popen([
                "python", "-m", "http.server", 
                "-d", "dashboards_basic", "8090"
            ])
            logger.info("✅ Dashboard server started on http://localhost:8090")
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard server: {str(e)}")
    
    # Start the main FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)