"""
FastAPI service for hotel booking cancellation prediction.

Endpoints
---------
GET  /health   -> {"status": "ok"}
POST /predict  -> {"prediction": 1, "label": "Cancelled", "cancellation_probability": 0.87}

Run:  uvicorn src.api:app --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_cancellation

app = FastAPI(
    title="Hotel Booking Cancellation Predictor",
    description="Predicts whether a booking will be cancelled and the probability.",
    version="1.0.0",
)


class BookingFeatures(BaseModel):
    """Booking input. Only send what you know — the rest use sensible defaults."""

    hotel: Optional[str] = Field(None, examples=["City Hotel", "Resort Hotel"])
    lead_time: Optional[int] = Field(None, ge=0, examples=[120])
    arrival_date_month: Optional[str] = Field(None, examples=["August"])
    adults: Optional[int] = Field(None, ge=0, examples=[2])
    children: Optional[int] = Field(None, ge=0, examples=[0])
    babies: Optional[int] = Field(None, ge=0, examples=[0])
    meal: Optional[str] = Field(None, examples=["BB"])
    country: Optional[str] = Field(None, examples=["PRT"])
    market_segment: Optional[str] = Field(None, examples=["Online TA"])
    distribution_channel: Optional[str] = Field(None, examples=["TA/TO"])
    is_repeated_guest: Optional[int] = Field(None, examples=[0])
    previous_cancellations: Optional[int] = Field(None, ge=0, examples=[0])
    previous_bookings_not_canceled: Optional[int] = Field(None, ge=0, examples=[0])
    reserved_room_type: Optional[str] = Field(None, examples=["A"])
    assigned_room_type: Optional[str] = Field(None, examples=["A"])
    booking_changes: Optional[int] = Field(None, ge=0, examples=[0])
    deposit_type: Optional[str] = Field(None, examples=["No Deposit", "Non Refund"])
    days_in_waiting_list: Optional[int] = Field(None, ge=0, examples=[0])
    customer_type: Optional[str] = Field(None, examples=["Transient"])
    adr: Optional[float] = Field(None, examples=[100.0])
    required_car_parking_spaces: Optional[int] = Field(None, ge=0, examples=[0])
    total_of_special_requests: Optional[int] = Field(None, ge=0, examples=[1])
    stays_in_weekend_nights: Optional[int] = Field(None, ge=0, examples=[1])
    stays_in_week_nights: Optional[int] = Field(None, ge=0, examples=[2])
    arrival_date_year: Optional[int] = Field(None, examples=[2016])
    arrival_date_week_number: Optional[int] = Field(None, examples=[27])
    arrival_date_day_of_month: Optional[int] = Field(None, examples=[15])
    agent: Optional[int] = Field(None, examples=[9])
    company: Optional[int] = Field(None, examples=[0])


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    cancellation_probability: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingFeatures) -> PredictionResponse:
    try:
        result = predict_cancellation(booking.model_dump(exclude_none=True))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - return clean error to caller
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result)
