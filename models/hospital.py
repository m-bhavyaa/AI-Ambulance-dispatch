"""
Hospital Data Model
Represents a hospital with capacity, ICU, and doctor availability.
Used as domain values in the CSP for hospital selection.
"""


class Hospital:
    """Represents a hospital in the emergency dispatch system."""

    def __init__(self, name, node, total_beds, available_beds, icu_available,
                 doctors_available, capacity_limit):
        """
        Initialize a Hospital.

        Args:
            name (str): Hospital name.
            node (str): Graph node where hospital is located.
            total_beds (int): Total number of beds.
            available_beds (int): Currently available beds.
            icu_available (bool): Whether ICU is available.
            doctors_available (int): Number of available doctors.
            capacity_limit (int): Maximum number of occupied beds the hospital
                will accept (a safe-operating / surge limit). The hospital can
                take a new patient only while occupied_beds < capacity_limit.
                It may be lower than total_beds, in which case this constraint
                is independent of bed availability.
        """
        self.name = name
        self.node = node
        self.total_beds = total_beds
        self.available_beds = available_beds
        self.icu_available = icu_available
        self.doctors_available = doctors_available
        self.capacity_limit = capacity_limit

    @property
    def occupied_beds(self):
        """Number of beds currently occupied."""
        return self.total_beds - self.available_beds

    def within_capacity_limit(self):
        """True if the hospital is still below its occupancy limit."""
        return self.occupied_beds < self.capacity_limit

    def admit_patient(self):
        """
        Admit a patient: reduce available beds and doctors by 1.
        Respects the same rules the CSP enforces (beds, doctors, capacity limit).
        """
        if self.has_capacity():
            self.available_beds -= 1
            self.doctors_available -= 1
            return True
        return False

    def discharge_patient(self):
        """Discharge a patient: increase available beds and doctors by 1."""
        if self.available_beds < self.total_beds:
            self.available_beds += 1
            self.doctors_available += 1
            return True
        return False

    def has_capacity(self):
        """Check if hospital can accept another patient."""
        return (self.available_beds > 0
                and self.doctors_available > 0
                and self.within_capacity_limit())

    def __repr__(self):
        return (f"Hospital({self.name}, beds={self.available_beds}/{self.total_beds}, "
                f"ICU={'Yes' if self.icu_available else 'No'}, "
                f"doctors={self.doctors_available})")
