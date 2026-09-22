from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConsensusTheme(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = ""
    description: str = ""
    evidence_by_country: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "title" not in mapped and "theme" in mapped:
                mapped["title"] = mapped["theme"]
            return mapped
        return data


class DisagreementPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    topic: str = ""
    description: str = ""
    stances_by_stakeholder: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "topic" not in mapped:
                mapped["topic"] = (
                    mapped.get("disagreement")
                    or mapped.get("title")
                    or mapped.get("name")
                    or mapped.get("point")
                    or "Strategic Divergence"
                )
            if "description" not in mapped:
                mapped["description"] = (
                    mapped.get("summary")
                    or mapped.get("details")
                    or mapped.get("content")
                    or str(mapped.get("disagreement", ""))
                )
            if "stances_by_stakeholder" not in mapped:
                mapped["stances_by_stakeholder"] = (
                    mapped.get("stances")
                    or mapped.get("stakeholder_stances")
                    or mapped.get("positions")
                    or mapped.get("stakeholders")
                    or {}
                )
            # Ensure stances values are strings
            if isinstance(mapped.get("stances_by_stakeholder"), dict):
                mapped["stances_by_stakeholder"] = {
                    str(k): str(v) for k, v in mapped["stances_by_stakeholder"].items()
                }
            return mapped
        return data


def _coerce_spectrum_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        if isinstance(item, dict):
            expert = item.get("expert") or item.get("market") or item.get("country") or ""
            outlook = item.get("growth") or item.get("outlook") or item.get("description") or str(item)
            result.append(f"{expert} — {outlook}" if expert else outlook)
        else:
            result.append(str(item))
    return result


class MarketGrowthSpectrum(BaseModel):
    model_config = ConfigDict(frozen=True)

    conservative: list[str] = Field(default_factory=list)
    moderate: list[str] = Field(default_factory=list)
    bullish: list[str] = Field(default_factory=list)

    @field_validator("conservative", "moderate", "bullish", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _coerce_spectrum_list(v)


class SynthesisReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_version: str = "synthesis_v1"
    source_matrix_version: str
    consensus_themes: list[ConsensusTheme] = Field(default_factory=list)
    disagreements: list[DisagreementPoint] = Field(default_factory=list)
    growth_spectrum: MarketGrowthSpectrum


__all__ = [
    "ConsensusTheme",
    "DisagreementPoint",
    "MarketGrowthSpectrum",
    "SynthesisReport",
]
