from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import ezkl
import json
import os
import base64
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prove")
async def generate_proof(input_data: dict):
    # 1. Save Input
    witness_data = {"input_data": [input_data["data"]]}
    with open("input.json", "w") as f:
        json.dump(witness_data, f)

    # 2. Generate Proof (uses the PRIVATE pk.key)
    # The server (Prover) holds the pk.key and model
    ezkl.gen_witness("input.json", "compiled.ezkl", "witness.json")
    ezkl.prove("witness.json", "compiled.ezkl", "pk.key", "proof.json", "kzg.srs")

    # 3. Read the Proof and the Public Key
    with open("proof.json", "r") as f:
        proof_content = json.load(f)
    
    # We read the vk.key as binary and encode it to base64 to send to frontend
    with open("vk.key", "rb") as f:
        vk_content = base64.b64encode(f.read()).decode('utf-8')

    # Return both to the "Uploader"
    return {
        "proof": proof_content, 
        "vk": vk_content
    }

@app.post("/verify")
async def verify_proof(
    proof_file: UploadFile = File(...), 
    vk_file: UploadFile = File(...)
):
    # 1. Save the UPLOADED files temporarily
    # This simulates the Verifier having NO access to the original server files
    # They strictly use what was uploaded.
    
    temp_proof_path = "temp_uploaded_proof.json"
    temp_vk_path = "temp_uploaded_vk.key"

    with open(temp_proof_path, "wb") as f:
        shutil.copyfileobj(proof_file.file, f)
        
    with open(temp_vk_path, "wb") as f:
        shutil.copyfileobj(vk_file.file, f)

    # 2. Verify using the UPLOADED keys
    # Note: settings.json and kzg.srs are considered "Public Parameters" 
    # (global infrastructure), so reading them from server is standard.
    try:
        res = ezkl.verify(
            temp_proof_path, 
            "settings.json", 
            temp_vk_path, 
            "kzg.srs"
        )
        return {"is_valid": res}
        
    except Exception as e:
        return {"is_valid": False, "error": str(e)}
    
    finally:
        # Cleanup temp files
        if os.path.exists(temp_proof_path): os.remove(temp_proof_path)
        if os.path.exists(temp_vk_path): os.remove(temp_vk_path)
