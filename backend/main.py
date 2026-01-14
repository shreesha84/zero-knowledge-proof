from fastapi import FastAPI
import ezkl
import json
import os

app = FastAPI()

@app.post("/prove")
async def generate_proof(input_data: dict):
    # input_data expected as {"data": [0.1, 0.2, 0.3]}
    
    # 1. Format for EZKL
    witness_data = {"input_data": [input_data["data"]]}
    with open("input.json", "w") as f:
        json.dump(witness_data, f)

    # 2. Generate Witness and Proof (Asynchronous)@app.post("/prove")
async def generate_proof(input_data: dict):
    # ... (file saving code) ...
    ezkl.gen_witness("input.json", "compiled.ezkl", "witness.json")
    ezkl.prove("witness.json", "compiled.ezkl", "pk.key", "proof.json", "single")
    # ...

    with open("proof.json", "r") as f:
        proof = json.load(f)

    return {"proof": proof}

@app.get("/verify")
async def verify_ownership():
    # 3. Verify the proof against the Verifying Key (Asynchronous)
    res = await ezkl.verify("proof.json", "settings.json", "vk.key")
    return {"is_valid": res}