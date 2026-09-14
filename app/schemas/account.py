"""Request/response models for the account endpoints (E5)."""
from pydantic import BaseModel, Field


class CreateAccountRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    plan: dict = Field(default_factory=dict)


class SignInRequest(BaseModel):
    username: str
    password: str


class SavePlanRequest(BaseModel):
    plan: dict = Field(default_factory=dict)


class AccountResponse(BaseModel):
    username: str
    display_name: str


class SignInResponse(BaseModel):
    username: str
    display_name: str
    plan: dict | None = None


class StatusResponse(BaseModel):
    status: str
