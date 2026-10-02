from datetime import datetime, timedelta
import unittest

from aquaflow.work.dispatch import (
    DispatchService,
    FatiguePolicy,
    WorkOrder,
)


START = datetime(2026, 10, 2, 9, 0)


def emergency_order():
    return {
        "id": "WO-1",
        "priority": 1,
        "starts_at": START.isoformat(),
        "confined_space": True,
        "lockout_tagout": True,
    }


def fully_qualified_crew():
    return {
        "id": "CREW-A",
        "members": [
            {"id": "A", "qualifications": ["crew_lead", "loto_authorized"]},
            {"id": "B", "qualifications": ["entrant", "confined_space"]},
            {"id": "C", "qualifications": ["attendant", "confined_space"]},
            {"id": "D", "qualifications": ["rescue", "confined_space"]},
        ],
    }


class DispatchFatigueTests(unittest.TestCase):
    def test_qualified_rested_crew_is_proposed(self):
        service = DispatchService(FatiguePolicy())
        history = {"A": 8.0, "B": 6.0, "C": 4.0, "D": 2.0}

        plan = service.propose(emergency_order(), [fully_qualified_crew()], history, START)
        evaluation = plan.options[0]

        self.assertEqual(plan.decision, "dispatch")
        self.assertEqual(plan.selected.crew_id, "CREW-A")
        self.assertTrue(evaluation.allowed)
        self.assertEqual(
            sorted(role for role, _ in evaluation.role_assignments),
            ["attendant", "crew_lead", "entrant", "loto_authorized", "rescue"],
        )
        self.assertEqual({row.worker_id for row in plan.fatigue_calculations}, {"A", "B", "C", "D"})
        self.assertTrue(all(not row.fatigued for row in plan.fatigue_calculations))
        self.assertEqual(len(service.fatigue_records), 4)
        self.assertTrue(any(entry.action == "crew_evaluated" for entry in service.audit_trail))

    def test_qualified_fatigued_crew_is_rejected(self):
        service = DispatchService()
        history = {"A": 13.0, "B": 13.0, "C": 13.0, "D": 13.0}
        original_history = dict(history)

        plan = service.propose(emergency_order(), [fully_qualified_crew()], history, START)
        evaluation = plan.options[0]

        self.assertEqual(plan.decision, "do_not_dispatch")
        self.assertIsNone(plan.selected)
        self.assertFalse(evaluation.allowed)
        self.assertIn("fatigued_worker", {item.code for item in evaluation.violations})
        self.assertTrue(all(row.fatigued for row in plan.fatigue_calculations))
        self.assertEqual(history, original_history)
        self.assertEqual([entry.decision for entry in service.audit_trail if entry.crew_id == "CREW-A"], ["denied"])

    def test_supervisor_override_expired_before_start_is_not_applied(self):
        service = DispatchService()
        history = {"A": 13.0, "B": 13.0, "C": 13.0, "D": 13.0}
        created = START - timedelta(hours=4)
        expires = START - timedelta(minutes=1)
        override = service.grant_override(
            "WO-1",
            "CREW-A",
            "supervisor.morgan",
            "Emergency repair must start, with standby rescue and fatigue monitoring.",
            created,
            expires,
            "OVR-1",
        )

        plan = service.propose(emergency_order(), [fully_qualified_crew()], history, START, override)
        evaluation = plan.options[0]
        audit_codes = {entry.action for entry in service.audit_trail}

        self.assertFalse(evaluation.allowed)
        self.assertFalse(evaluation.override_applied)
        self.assertIn("expired_override", {item.code for item in evaluation.violations})
        self.assertIn("supervisor_override_granted", audit_codes)
        self.assertIn("supervisor_override_expired", audit_codes)
        grant = next(entry for entry in service.audit_trail if entry.action == "supervisor_override_granted")
        self.assertEqual(grant.reason, override.reason)
        self.assertEqual(grant.expires_at, expires)
        self.assertEqual(history, {"A": 13.0, "B": 13.0, "C": 13.0, "D": 13.0})

    def test_active_supervisor_override_allows_only_fatigue(self):
        service = DispatchService()
        history = {"A": 13.0, "B": 6.0, "C": 4.0, "D": 2.0}
        original_history = dict(history)
        override = service.grant_override(
            "WO-1",
            "CREW-A",
            "supervisor.morgan",
            "Emergency response authorized with fatigue checks at the site.",
            START - timedelta(minutes=30),
            START + timedelta(hours=1),
        )

        plan = service.propose(emergency_order(), [fully_qualified_crew()], history, START, override)

        self.assertEqual(plan.decision, "dispatch")
        self.assertEqual(plan.selected.crew_id, "CREW-A")
        self.assertTrue(plan.override_applied)
        self.assertEqual(plan.override.id, override.id)
        self.assertTrue(any(entry.action == "supervisor_override_applied" for entry in service.audit_trail))
        self.assertEqual(history, original_history)

    def test_active_override_does_not_bypass_unsafe_role_combination(self):
        service = DispatchService()
        unsafe_crew = {
            "id": "CREW-UNSAFE",
            "members": [
                {"id": "A", "qualifications": ["crew_lead", "loto_authorized"]},
                # Only one confined-space worker means entrant/attendant/rescue
                # cannot be filled by three distinct people.
                {"id": "B", "qualifications": ["entrant", "attendant", "rescue", "confined_space"]},
            ],
        }
        override = service.grant_override(
            "WO-1",
            "CREW-UNSAFE",
            "supervisor.morgan",
            "Attempted emergency acceptance.",
            START - timedelta(hours=1),
            START + timedelta(hours=1),
        )

        plan = service.propose(emergency_order(), [unsafe_crew], {"A": 13.0, "B": 13.0}, START, override)

        self.assertEqual(plan.decision, "do_not_dispatch")
        self.assertIn("unsafe_role_combination", {item.code for item in plan.options[0].violations})
        self.assertFalse(plan.options[0].override_applied)


if __name__ == "__main__":
    unittest.main()
