from fastapi import FastAPI
import ezkl
import json

app = FastAPI()

@app.post("/prove")
async def generate_proof(input_data: list):
    
    witness_data = {"input_data": [input_data]}
    with open("input.json", "w") as f:
        json.dump(witness_data, f)

    # 2. Generate Witness (Calculates the model output internally)
    ezkl.gen_witness("input.json", "compiled.ezkl", "witness.json")

    # 3. Create the Proof
    # This uses the Proving Key to create a ZK-SNARK
    ezkl.prove("witness.json", "compiled.ezkl", "pk.key", "proof.json", "single")

    with open("proof.json", "r") as f:
        proof = json.load(f)

    return {"proof": proof}

@app.get("/verify")
async def verify_ownership():
    # Verifies the generated proof against the Public Verifying Key
    res = ezkl.verify("proof.json", "settings.json", "vk.key")
    return {"is_valid": res}