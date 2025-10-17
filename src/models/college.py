from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class VerificationStatus(Enum):
    DRAFT = "Draft"
    VERIFIED = "Verified"
    PUBLISHED = "Published"

class EvidenceStatus(Enum):
    VERIFIED = "Verified"
    PARTIALLY_VERIFIED = "Partially Verified"
    NO_EVIDENCE_FOUND = "No Evidence Found"
    PENDING_VERIFICATION = "Pending Verification"

class DegreeLevel(Enum):
    UG = "Undergradudate"
    PG = "Postgraduate"
    DIPLOMA = "Diploma"
    CERTIFICATE = "Certificate"
    PHD = "PhD"

@dataclass
class Course:
    name: str
    # degree_level: DegreeLevel
    degree_level: str
    official_source_url: str
    row_confidence: float
    duration: Optional[str] = None
    annual_fees: Optional[str] = None
    seats: Optional[int] = None
    entrance_exams: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    evidence_urls: List[str] = field(default_factory=list)

@dataclass
class College:
    name: str
    city: str
    state: str
    type: str
    website: str
    overall_confidence: float
    last_collected: datetime
    verification_status: VerificationStatus
    evidence_status: EvidenceStatus
    courses: List[Course] = field(default_factory=list)
    evidence_urls: List[str] = field(default_factory=list)