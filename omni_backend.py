from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
import shutil
import time
from PIL import Image
import numpy as np
import logging
import torch
from io import BytesIO  # Moved to top for proper usage

# Configure logging FIRST (critical for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle pkg_resources deprecation BEFORE importing cellpose_omni
try:
    # Try to avoid pkg_resources usage
    import importlib.metadata
    try:
        version = importlib.metadata.version("cellpose_omni")
        logger.info(f"✅ cellpose_omni version: {version}")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("⚠️ Could not determine cellpose_omni version")
        
    # Import cellpose_omni
    from cellpose_omni import io, models
    logger.info("✅ Successfully imported cellpose_omni")
    
except Exception as e:
    logger.error(f"❌ Critical import error: {str(e)}")
    # Try fallback imports
    try:
        logger.info("⚠️ Attempting fallback to standard cellpose")
        from cellpose import models
        from omnipose import utils
        logger.warning("⚠️ Using standard cellpose instead of cellpose_omni")
    except Exception as e:
        logger.error(f"❌ Failed to import fallback packages: {str(e)}")
        raise

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to save uploads and results
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Clear existing models to force re-download (critical for Mac)
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cellpose", "models")
if os.path.exists(MODEL_DIR):
    logger.info(f"Clearing existing models directory: {MODEL_DIR}")
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("✅ Successfully cleared model directory")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear model directory: {str(e)}")

# Load Omnipose model once at startup
logger.info("Loading Omnipose model... This may take a moment")

model = None
MAX_RETRIES = 5  # Increased retries
RETRY_DELAY = 3  # Increased delay

# STRICTLY USE cyto2 - NO FALLBACKS
for retry in range(MAX_RETRIES):
    try:
        logger.info(f"Attempt {retry+1}/{MAX_RETRIES}: Loading cyto2 model with CPU mode")
        
        # Create model instance - STRICT cyto2_omni
        model = models.CellposeModel(
            gpu=False,
            model_type="cyto2_omni"  # CRITICAL: Use cyto2_omni, not cyto2
        )
        
        # Verify model loaded correctly
        if model.pretrained_model is None:
            raise ValueError("Model pretrained_model attribute is None")
            
        logger.info(f"✅ Successfully loaded cyto2_omni model (CPU mode)")
        logger.info(f"Model details: type={type(model).__name__}, pretrained_model={model.pretrained_model}")
        break  # Success! Exit the retry loop
        
    except Exception as e:
        logger.error(f"❌ Error loading cyto2_omni model (attempt {retry+1}): {str(e)}")
        
        # Additional diagnostics for the specific error
        if "central directory" in str(e) or "zip archive" in str(e):
            logger.warning("⚠️ This is likely a timing issue with model file download, not a corrupted file")
            model_path = os.path.join(MODEL_DIR, "cyto2torch_0")
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / 1024
                logger.info(f"Model file exists at: {model_path} (size: {file_size:.2f} KB)")
                # Check if file size is reasonable (should be ~25MB)
                if file_size < 20000:  # Less than 20MB likely incomplete
                    logger.warning("⚠️ Model file appears incomplete - deleting for fresh download")
                    try:
                        os.remove(model_path)
                        logger.info("✅ Deleted incomplete model file")
                    except Exception as e:
                        logger.error(f"❌ Could not delete incomplete model file: {str(e)}")
        
        if retry < MAX_RETRIES - 1:
            logger.info(f"⏳ Waiting {RETRY_DELAY} seconds before retry (to allow file system to settle)...")
            time.sleep(RETRY_DELAY)
        else:
            logger.critical("❌ CRITICAL: Failed to load cyto2_omni model after multiple attempts")
            # No fallback - we strictly want cyto2
            raise RuntimeError("cyto2_omni model failed to load") from e

@app.post("/upload-and-segment")
async def upload_and_segment(file: UploadFile = File(...)):
    if model is None:
        logger.critical("CRITICAL: Model failed to load during startup - cyto2_omni required")
        raise HTTPException(status_code=500, detail="Segmentation service is unavailable (cyto2_omni model failed to load)")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            logger.warning(f"Invalid file type attempted: {file.content_type}")
            raise HTTPException(status_code=400, detail="Invalid image format. Please upload a PNG, JPG, or TIFF file.")
        
        logger.info(f"Processing image: {file.filename} ({file.content_type})")
        
        # Read image
        contents = await file.read()
        try:
            # CORRECTED: Use BytesIO to handle the image bytes properly
            img_pil = Image.open(BytesIO(contents))
            
            # Handle different image modes properly for cyto2
            if img_pil.mode == 'L':  # Already grayscale
                img_np = np.array(img_pil)
            elif img_pil.mode == 'RGB' or img_pil.mode == 'RGBA':
                # Convert to grayscale while preserving dtype
                img_np = np.array(img_pil.convert('L'))
            else:
                # Convert any other mode to RGB then to grayscale
                img_np = np.array(img_pil.convert('RGB').convert('L'))
            
            logger.info(f"Image shape after conversion: {img_np.shape}, dtype: {img_np.dtype}")
            
            # Ensure image is in the correct format for cyto2
            if img_np.ndim == 2:
                # cyto2 expects 2D images (Y, X), not 3D
                pass
            elif img_np.ndim == 3:
                if img_np.shape[-1] == 3 or img_np.shape[-1] == 4:
                    # RGB or RGBA to grayscale (already done above, but double-check)
                    img_np = np.mean(img_np[..., :3], axis=-1).astype(np.uint8)
                else:
                    logger.error(f"Unexpected image dimensions after conversion: {img_np.shape}")
                    raise HTTPException(status_code=400, detail="Unsupported image format")
            else:
                logger.error(f"Image has unexpected number of dimensions: {img_np.ndim}")
                raise HTTPException(status_code=400, detail="Unsupported image format")
                
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file. Could not open the image.")
        
        # Run Omnipose segmentation with CORRECT parameters for cyto2
        logger.info("Starting Omnipose segmentation with cyto2_omni...")
        
        masks, flows, styles, diams = model.eval(
            img_np,
            channels=[0, 0],      # Gray image
            omni=True,            # REQUIRED for Omnipose
            invert=False,
            diameter=15,
            mask_threshold=0.0,   # Use mask_threshold as in working code
            flow_threshold=0.45,  # cyto2 works better with this value
            do_3D=False,
            tile=False
        )
        
        logger.info(f"Segmentation completed successfully. Masks shape: {masks.shape}, max value: {masks.max()}")
        logger.info(f"Detected {int(np.max(masks))} cells")
        
        # Save results
        out_id = str(uuid.uuid4())
        logger.info(f"Creating result directory: {os.path.join(RESULT_DIR, out_id)}")
        result_subdir = os.path.join(RESULT_DIR, out_id)
        os.makedirs(result_subdir, exist_ok=True)
        
        # Save masks
        files = [os.path.join(UPLOAD_DIR, f"{out_id}_input.png")]
        logger.info(f"Saving masks to: {result_subdir}")
        io.save_masks(
            [img_np],
            [masks],
            [flows],
            files,
            tif=True,
            png=False,
            in_folders=True,
            save_flows=True,
            save_outlines=True,
            save_ncolor=True,
            savedir=result_subdir
        )
        
        # Return response
        base_url = "http://localhost:8000"
        return {
            "id": out_id,
            "message": "Segmentation completed successfully",
            "result_url": f"{base_url}/result/{out_id}/masks_ncolor_0.png",
            "mask_url": f"{base_url}/mask/{out_id}",
            "max_mask_value": int(masks.max()),
            "mask_shape": masks.shape,
            "mask_type": str(masks.dtype),
            "mask_count": int(np.max(masks))
        }
    
    except Exception as e:
        logger.exception("Error during image processing")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/result/{out_id}")
async def get_result_image(out_id: str):
    ncolor_path = os.path.join(RESULT_DIR, out_id, f"masks_ncolor_0.png")
    if os.path.exists(ncolor_path):
        logger.info(f"Serving result image from: {ncolor_path}")
        return FileResponse(ncolor_path, media_type="image/png")
    
    path = os.path.join(RESULT_DIR, f"{out_id}_result.png")
    if os.path.exists(path):
        logger.info(f"Serving fallback result image from: {path}")
        return FileResponse(path, media_type="image/png")
    
    logger.error(f"Result image not found for ID: {out_id}")
    raise HTTPException(status_code=404, detail="Result image not found")

@app.get("/mask/{out_id}")
async def get_mask_json(out_id: str):
    mask_path = os.path.join(RESULT_DIR, out_id, "masks_0.npy")
    if os.path.exists(mask_path):
        logger.info(f"Serving mask data from: {mask_path}")
        masks = np.load(mask_path)
        return {
            "id": out_id,
            "max_mask_value": int(masks.max()),
            "mask_shape": masks.shape,
            "mask_type": str(masks.dtype),
            "mask_count": int(np.max(masks))
        }
    
    logger.error(f"Mask file not found: {out_id}")
    raise HTTPException(status_code=404, detail="Mask data not found")

if __name__ == "__main__":
    logger.info("🚀 Starting Omnipose backend server at http://localhost:8000")
    logger.info("💡 Make sure to run 'python -m http.server 8080' in another terminal for the frontend")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)