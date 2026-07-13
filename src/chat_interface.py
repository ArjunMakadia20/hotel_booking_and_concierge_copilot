"""
Interactive terminal chat interface for single-booking cancellation inference.

A local demo/dev tool (not the production API). It runs a REPL loop that, on each
turn, lets the user choose how to enter one booking:

- **Guided mode** prompts one field at a time, showing the default and (for known
  categoricals) the valid options. Pressing Enter accepts the documented default.
- **JSON mode** accepts a single pasted JSON object; missing optional fields fall
  back to the same defaults as guided mode.

Either way the collected booking is scored through the canonical serving pipeline via
:func:`src.predict.predict_cancellation` (no preprocessing is reimplemented here) and
the predicted class, cancellation probability and a plain-language risk summary are
printed. ``exit``/``quit`` ends the session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.predict import FEATURE_DEFAULTS, predict_cancellation

logger = logging.getLogger(__name__)

QUIT_WORDS = {"exit", "quit"}

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "meal": ["BB", "FB", "HB", "SC", "Undefined"],
    "market_segment": [
        "Aviation", "Complementary", "Corporate", "Direct",
        "Groups", "Offline TA/TO", "Online TA", "Undefined",
    ],
    "distribution_channel": ["Corporate", "Direct", "GDS", "TA/TO", "Undefined"],
    "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Contract", "Group", "Transient", "Transient-Party"],
}


@dataclass(frozen=True)
class Field:
    """One collectable booking field with its validation/coercion behaviour.

    ``kind`` is one of ``"int"``, ``"float"``, ``"str"`` or ``"category"``. ``options``
    lists the valid values for ``"category"`` fields; ``help_text`` is an optional hint
    shown alongside the guided-mode prompt.
    """

    name: str
    kind: str
    options: list[str] | None = None
    help_text: str = ""


FIELDS: list[Field] = [
    Field("hotel", "category", CATEGORICAL_OPTIONS["hotel"]),
    Field("lead_time", "int", help_text="days between booking and arrival"),
    Field("arrival_date_month", "category", CATEGORICAL_OPTIONS["arrival_date_month"]),
    Field("arrival_date_year", "int"),
    Field("arrival_date_week_number", "int", help_text="1-53"),
    Field("arrival_date_day_of_month", "int", help_text="1-31"),
    Field("stays_in_weekend_nights", "int"),
    Field("stays_in_week_nights", "int"),
    Field("adults", "int"),
    Field("children", "int"),
    Field("babies", "int"),
    Field("meal", "category", CATEGORICAL_OPTIONS["meal"]),
    Field("country", "str", help_text="ISO code, e.g. PRT, GBR, USA"),
    Field("market_segment", "category", CATEGORICAL_OPTIONS["market_segment"]),
    Field("distribution_channel", "category", CATEGORICAL_OPTIONS["distribution_channel"]),
    Field("is_repeated_guest", "int", help_text="0 or 1"),
    Field("previous_cancellations", "int"),
    Field("previous_bookings_not_canceled", "int"),
    Field("reserved_room_type", "category", CATEGORICAL_OPTIONS["reserved_room_type"]),
    Field("deposit_type", "category", CATEGORICAL_OPTIONS["deposit_type"]),
    Field("days_in_waiting_list", "int"),
    Field("customer_type", "category", CATEGORICAL_OPTIONS["customer_type"]),
    Field("adr", "float", help_text="average daily rate"),
    Field("required_car_parking_spaces", "int"),
    Field("total_of_special_requests", "int"),
    Field("agent", "int", help_text="travel agent ID; blank/0 -> NONE"),
    Field("company", "int", help_text="company ID; blank/0 -> NONE"),
]

FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
REQUIRED_JSON_FIELDS: set[str] = set()


class ValidationError(ValueError):
    """Raised when a supplied value is not valid for its field."""


def _coerce_value(field: Field, raw: str) -> Any:
    """Coerce and validate a raw string for ``field``; raise on invalid input."""
    text = raw.strip()
    if field.kind == "int":
        try:
            return int(text)
        except ValueError as exc:
            raise ValidationError(f"'{field.name}' must be a whole number, got '{raw}'.") from exc
    if field.kind == "float":
        try:
            return float(text)
        except ValueError as exc:
            raise ValidationError(f"'{field.name}' must be a number, got '{raw}'.") from exc
    if field.kind == "category":
        match = _match_option(field, text)
        if match is None:
            raise ValidationError(
                f"'{field.name}' must be one of {field.options}, got '{raw}'."
            )
        return match
    return text


def _match_option(field: Field, text: str) -> str | None:
    """Return the canonical option matching ``text`` case-insensitively, or None."""
    if not field.options:
        return text
    for option in field.options:
        if option.lower() == text.lower():
            return option
    return None


def validate_booking(booking: dict[str, Any]) -> dict[str, Any]:
    """Validate/coerce a raw booking dict (used by JSON mode) against the field specs.

    Unknown keys are ignored (the serving layer only reads known features). Category
    values are matched case-insensitively to their canonical form. Numeric fields are
    coerced to int/float. Raises :class:`ValidationError` on any bad value or missing
    required field.
    """
    missing = REQUIRED_JSON_FIELDS - set(booking)
    if missing:
        raise ValidationError(f"Missing required field(s): {sorted(missing)}.")

    cleaned: dict[str, Any] = {}
    for key, value in booking.items():
        field = FIELDS_BY_NAME.get(key)
        if field is None:
            logger.debug("ignoring unknown field '%s'", key)
            continue
        if value is None or value == "":
            continue
        cleaned[key] = _coerce_value(field, str(value))
    return cleaned


def _default_hint(field: Field) -> str:
    """Human-readable default label for a field's guided-mode prompt."""
    default = FEATURE_DEFAULTS[field.name]
    if field.name in ("agent", "company") and default == 0:
        return "NONE"
    return str(default)


def collect_guided(prompt: Callable[[str], str]) -> dict[str, Any]:
    """Prompt for each field in turn, returning the collected booking dict.

    ``prompt`` is an injectable input function (defaults to :func:`input` in the CLI).
    Pressing Enter accepts the documented default. Invalid values are rejected with a
    message and the same field is asked again.
    """
    booking: dict[str, Any] = {}
    for field in FIELDS:
        hint_parts = [f"default: {_default_hint(field)}"]
        if field.help_text:
            hint_parts.insert(0, field.help_text)
        if field.options:
            hint_parts.insert(0, "options: " + ", ".join(field.options))
        label = f"{field.name} [{'; '.join(hint_parts)}]: "

        while True:
            raw = prompt(label).strip()
            if raw == "":
                break
            try:
                booking[field.name] = _coerce_value(field, raw)
                break
            except ValidationError as exc:
                print(f"  ! {exc} Try again or press Enter for the default.")
    return booking


def parse_json_booking(text: str) -> dict[str, Any]:
    """Parse and validate a pasted JSON object into a booking dict.

    Raises :class:`ValidationError` on malformed JSON, a non-object payload, or any
    invalid field value.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Malformed JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValidationError("JSON must be a single object, e.g. {\"lead_time\": 120}.")
    return validate_booking(payload)


def format_prediction(result: dict[str, Any]) -> str:
    """Render a prediction dict as a multi-line terminal report with a risk summary."""
    proba = result["cancellation_probability"]
    percent = proba * 100
    if proba >= 0.8:
        band = "High risk"
    elif proba >= 0.5:
        band = "Elevated risk"
    elif proba >= 0.2:
        band = "Low risk"
    else:
        band = "Very low risk"
    verb = "will cancel" if result["prediction"] == 1 else "will not cancel"
    return (
        f"  Predicted class : {result['label']} ({result['prediction']})\n"
        f"  Cancel probability: {proba:.4f}\n"
        f"  Summary          : {band} - {percent:.1f}% probability of cancellation "
        f"(model predicts the booking {verb})."
    )


def _read_json_block(prompt: Callable[[str], str]) -> str:
    """Read one or more lines of JSON until braces balance or a blank line ends it."""
    lines: list[str] = []
    while True:
        line = prompt("json> " if not lines else "...  ")
        if line.strip() == "" and lines:
            break
        lines.append(line)
        joined = "\n".join(lines)
        if joined.count("{") > 0 and joined.count("{") == joined.count("}"):
            break
    return "\n".join(lines)


def run_once(prompt: Callable[[str], str]) -> bool:
    """Run one interaction turn. Returns False when the user asks to quit.

    Offers the mode menu, collects a booking, scores it and prints the report. Any
    error in collection or scoring is caught and reported without ending the session.
    """
    choice = prompt("\nMode - (g)uided, (j)son, or (q)uit: ").strip().lower()
    if choice in QUIT_WORDS or choice == "q":
        return False

    try:
        if choice in ("g", "guided"):
            booking = collect_guided(prompt)
        elif choice in ("j", "json"):
            print("Paste a JSON booking object (blank line to finish):")
            booking = parse_json_booking(_read_json_block(prompt))
        else:
            print("  ! Please enter 'g', 'j', or 'q'.")
            return True

        result = predict_cancellation(booking)
        logger.info(
            "scored booking: label=%s proba=%.4f",
            result["label"],
            result["cancellation_probability"],
        )
        print(format_prediction(result))
    except ValidationError as exc:
        print(f"  ! {exc}")
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive on any failure
        logger.exception("prediction failed")
        print(f"  ! Prediction failed: {exc}")
    return True


def run_chat(prompt: Callable[[str], str] = input) -> None:
    """Run the interactive prediction loop until the user quits.

    ``prompt`` is injectable so the loop can be driven in tests; it defaults to the
    built-in :func:`input`.
    """
    print("Hotel Booking Cancellation - interactive predictor")
    print("Enter a booking each turn; type 'q'/'exit'/'quit' at the menu to leave.")
    while True:
        try:
            if not run_once(prompt):
                break
        except (EOFError, KeyboardInterrupt):
            break
    print("\nGoodbye.")
