"""Part D: guarded FastAPI fraud triage service."""
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from prepare import add_features
from casenote import run_case

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "service.log"
logging.basicConfig(
    filename=str(LOG_PATH), level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("fraud_triage")
app = FastAPI(title="Merchant Risk Desk Fraud Triage", version="1.0.0")
metrics = {"requests": 0, "success": 0, "rejected": 0, "errors": 0}
_bundle = None


class TriageResponse(BaseModel):
    txn_id: str
    fraud_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    decision: str
    case_note: Optional[dict] = None
    policy_id: Optional[str] = None
    case_note_attempts: Optional[int] = None
    case_note_status: Optional[str] = None
    case_note_error: Optional[str] = None

    @field_validator("decision")
    @classmethod
    def valid_decision(cls, value):
        if value not in {"approve", "hold_for_review"}:
            raise ValueError("invalid decision")
        return value


class TransactionRequest(BaseModel):
    txn_id: str = Field(min_length=1, max_length=32)
    merchant_category: str = Field(min_length=1)
    city: str = Field(min_length=1)
    payment_mode: str = Field(min_length=1)
    amount_inr: float = Field(gt=0, le=1_000_000)
    hour_of_day: int = Field(ge=0, le=23)
    txn_count_last_24h: int = Field(ge=0, le=100)
    avg_amount_30d_inr: float = Field(gt=0)
    distance_from_home_km: float = Field(ge=0, le=100_000)
    is_new_device: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    question: str = Field(min_length=3, max_length=500)
    fraud_probability: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("payment_mode")
    @classmethod
    def valid_mode(cls, value):
        allowed = {"upi", "card", "netbanking", "wallet"}
        if value.strip().lower() not in allowed:
            raise ValueError("payment_mode must be UPI, Card, NetBanking or Wallet")
        return value.strip().title() if value.strip().lower() != "upi" else "UPI"


@app.middleware("http")
async def request_logging(request: Request, call_next):
    metrics["requests"] += 1
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            metrics["rejected"] += 1
        else:
            metrics["success"] += 1
        logger.info("method=%s path=%s status=%s", request.method, request.url.path, response.status_code)
        return response
    except Exception:
        metrics["errors"] += 1
        logger.exception("Unhandled service error path=%s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal service error"})


@app.on_event("startup")
def load_model():
    global _bundle
    model_path = ROOT / "model_bundle.pkl"
    if model_path.exists():
        with model_path.open("rb") as f:
            _bundle = pickle.load(f)
    else:
        logger.warning("model_bundle.pkl missing; only explicit fraud_probability can be used")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.get("/metrics")
def get_metrics():
    return dict(metrics)


@app.post("/triage", response_model=TriageResponse)
def triage(payload: TransactionRequest):
    try:
        if payload.fraud_probability is not None:
            probability = float(payload.fraud_probability)
        else:
            if _bundle is None:
                raise HTTPException(status_code=503, detail="model not trained; run python train.py")
            row = pd.DataFrame([payload.model_dump()])
            row = add_features(row)
            probability = float(_bundle["model"].predict_proba(row[_bundle["features"]])[:, 1][0])
        threshold = float(_bundle["threshold"]) if _bundle else 0.5
        flagged = probability >= threshold
        response = {
            "txn_id": payload.txn_id,
            "fraud_probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "decision": "hold_for_review" if flagged else "approve",
            "case_note": None,
        }
        if flagged:
            case = run_case(payload.txn_id, probability, payload.amount_inr, payload.question)
            if case["status"] == "ok":
                response["case_note"] = case["case_note"]
                response["policy_id"] = case["policy_id"]
                response["case_note_attempts"] = case["attempts"]
            elif case["status"] == "error":
                # Fail closed: do not approve if the note service is unavailable.
                response["decision"] = "hold_for_review"
                response["case_note_status"] = "unavailable"
                response["case_note_error"] = case["error"]
            else:
                response["case_note_status"] = case["status"]
                response["case_note_error"] = case["error"]
        logger.info("txn_id=%s probability=%.4f decision=%s", payload.txn_id, probability, response["decision"])
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("triage failed txn_id=%s", payload.txn_id)
        raise HTTPException(status_code=500, detail="triage failed safely")
