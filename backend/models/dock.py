from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

class VehicleStatus(str, Enum):
    PLACED = "placed"
    NOT_PLACED = "not_placed"

class HumanPresence(str, Enum):
    PRESENT = "present"
    NOT_PRESENT = "not_present"

class DockStatus(str, Enum):
    IDLE = "idle"
    OCCUPIED = "occupied"
    MAINTENANCE = "maintenance"

class DockData(BaseModel):
    dock_id: str
    vehicle_status: VehicleStatus
    human_presence: HumanPresence
    dock_status: DockStatus
    ideal_time_minutes: int = 0
    non_ideal_time_minutes: int = 0
    last_updated: datetime = datetime.now()
    notes: Optional[str] = None

class DockUpdate(BaseModel):
    vehicle_status: Optional[VehicleStatus] = None
    human_presence: Optional[HumanPresence] = None
    dock_status: Optional[DockStatus] = None
    notes: Optional[str] = None
