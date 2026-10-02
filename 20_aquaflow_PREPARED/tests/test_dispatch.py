import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.api import AquaFlowService
from aquaflow.work.dispatch import (
    CrewMember,
    DispatchBoard,
    DutyEntry,
    Override,
    WorkOrder,
)

UTC = timezone.utc
START = datetime(2026, 10, 2, 8, 0, tzinfo=UTC)


def order(**kw):
    base = dict(
        order_id="WO-1",
        priority=1,
        required_skills=frozenset({"pipe-repair"}),
        starts=START,
        estimated_hours=4.0,
        confined_space=True,
        lockout_tagout=True,
    )
    base.update(kw)
    return WorkOrder(**base)


def crew():
    return [
        CrewMember("A", frozenset({"pipe-repair", "lockout"})),
        CrewMember("B", frozenset({"pipe-repair"})),
        CrewMember("C", frozenset({"pipe-repair", "rescue"})),
    ]


def fatigue_b(board):
    # B worked 10h and stopped only 2h before the work starts.
    board.record_duty(DutyEntry("B", START - timedelta(hours=12), START - timedelta(hours=2)))


class TestQualifiedRestedCrew(unittest.TestCase):
    def test_assigns_with_distinct_roles_and_persists_fatigue(self):
        board = DispatchBoard()
        board.record_duty(DutyEntry("A", START - timedelta(days=2, hours=4), START - timedelta(days=2)))
        decision = board.propose(order(), crew(), at=START - timedelta(hours=1))
        self.assertEqual(decision.decision, "assign")
        self.assertEqual(decision.violations, ())
        self.assertEqual(len(decision.crew), 3)
        roles = {a.crew_id: a.role for a in decision.crew}
        self.assertEqual(
            {roles["A"], roles["B"], roles["C"]},
            {"entrant", "attendant", "rescue"},
        )
        # the fatigue calculation is persisted on the decision
        self.assertEqual(len(decision.fatigue), 3)
        self.assertTrue(all(not r.fatigued for r in decision.fatigue))
        # ... and in the audit trail under the same sequence number
        entries = [e for e in board.audit_log() if e["kind"] == "decision"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["seq"], decision.audit_seq)
        self.assertEqual(len(entries[0]["fatigue"]), 3)
        self.assertEqual(board.decisions(), (decision,))


class TestQualifiedButFatiguedCrew(unittest.TestCase):
    def test_rejects_and_persists_fatigue_calculation(self):
        board = DispatchBoard()
        fatigue_b(board)
        decision = board.propose(order(), crew(), at=START - timedelta(hours=1))
        self.assertEqual(decision.decision, "reject")
        self.assertTrue(any(v.startswith("fatigue:B") for v in decision.violations))
        report = {r.crew_id: r for r in decision.fatigue}["B"]
        self.assertTrue(report.fatigued)
        self.assertEqual(report.hours_24h, 10.0)
        self.assertEqual(report.rest_hours, 2.0)
        self.assertIn("daily_hours_exceeded", report.reasons)
        self.assertIn("insufficient_rest", report.reasons)
        # persisted: both the decision log and the audit entry carry the math
        self.assertEqual(board.decisions(), (decision,))
        entry = [e for e in board.audit_log() if e["kind"] == "decision"][0]
        snap = {f["crew_id"]: f for f in entry["fatigue"]}["B"]
        self.assertTrue(snap["fatigued"])
        self.assertEqual(snap["hours_24h"], 10.0)


class TestExpiredOverride(unittest.TestCase):
    def test_override_expired_before_work_starts_rejects_and_audits(self):
        board = DispatchBoard()
        fatigue_b(board)
        history_before = board.duty_history()
        board.add_override(Override(
            override_id="OV-1",
            order_id="WO-1",
            crew_id="B",
            supervisor="sup-1",
            reason="emergency staffing",
            created_at=START - timedelta(hours=6),
            expires_at=START - timedelta(hours=1),  # expires before work starts
        ))
        decision = board.propose(order(), crew(), at=START - timedelta(minutes=30))
        self.assertEqual(decision.decision, "reject")
        self.assertIn("override_expired:B", decision.violations)
        self.assertEqual(decision.overrides_applied, ())
        # the override itself is on the audit record
        kinds = [e["kind"] for e in board.audit_log()]
        self.assertEqual(kinds, ["override", "decision"])
        entry = board.audit_log()[0]
        self.assertEqual(entry["override_id"], "OV-1")
        self.assertEqual(entry["supervisor"], "sup-1")
        self.assertEqual(entry["reason"], "emergency staffing")
        # the override never touched the original duty history
        self.assertEqual(board.duty_history(), history_before)


class TestValidOverride(unittest.TestCase):
    def test_active_override_assigns_without_rewriting_history(self):
        board = DispatchBoard()
        fatigue_b(board)
        history_before = board.duty_history()
        board.add_override(Override(
            override_id="OV-9",
            order_id="WO-1",
            crew_id="B",
            supervisor="sup-7",
            reason="storm response staffing",
            created_at=START - timedelta(hours=1),
            expires_at=START + timedelta(hours=6),
        ))
        decision = board.propose(order(), crew())
        self.assertEqual(decision.decision, "assign")
        self.assertEqual(decision.overrides_applied, ("OV-9",))
        # the fatigue calculation is preserved, not rewritten by the override
        report = {r.crew_id: r for r in decision.fatigue}["B"]
        self.assertTrue(report.fatigued)
        self.assertEqual(report.hours_24h, 10.0)
        self.assertEqual(board.duty_history(), history_before)
        self.assertEqual([e["kind"] for e in board.audit_log()], ["override", "decision"])


class TestUnsafeRoleCombinations(unittest.TestCase):
    def test_confined_space_requires_three_distinct_crew(self):
        board = DispatchBoard()
        decision = board.propose(order(), crew()[:2])
        self.assertEqual(decision.decision, "reject")
        self.assertIn("confined_space_requires_3_distinct_crew", decision.violations)

    def test_lockout_requires_qualified_member(self):
        board = DispatchBoard()
        members = [CrewMember(c, frozenset({"pipe-repair"})) for c in "ABC"]
        decision = board.propose(order(), members)
        self.assertEqual(decision.decision, "reject")
        self.assertIn("lockout_qualified_crew_required", decision.violations)

    def test_missing_skills_rejected(self):
        board = DispatchBoard()
        members = [
            CrewMember("A", frozenset({"lockout"})),
            CrewMember("B", frozenset()),
            CrewMember("C", frozenset()),
        ]
        decision = board.propose(order(), members)
        self.assertEqual(decision.decision, "reject")
        self.assertIn("missing_skills:pipe-repair", decision.violations)

    def test_override_cannot_fix_role_safety(self):
        board = DispatchBoard()
        board.add_override(Override(
            override_id="OV-2",
            order_id="WO-1",
            crew_id="A",
            supervisor="sup-1",
            reason="urgent",
            created_at=START - timedelta(hours=1),
            expires_at=START + timedelta(hours=2),
        ))
        decision = board.propose(order(), crew()[:2])
        self.assertEqual(decision.decision, "reject")
        self.assertIn("confined_space_requires_3_distinct_crew", decision.violations)


class TestOverrideRules(unittest.TestCase):
    def test_override_requires_reason_and_future_expiry(self):
        board = DispatchBoard()
        with self.assertRaises(ValueError):
            board.add_override(Override(
                "OV-3", "WO-1", "B", "sup-1", "  ",
                START, START + timedelta(hours=1),
            ))
        with self.assertRaises(ValueError):
            board.add_override(Override("OV-4", "WO-1", "B", "sup-1", "ok", START, START))
        self.assertEqual(board.audit_log(), ())

    def test_override_only_honored_for_emergency_priority(self):
        board = DispatchBoard()
        fatigue_b(board)
        board.add_override(Override(
            override_id="OV-5",
            order_id="WO-1",
            crew_id="B",
            supervisor="sup-1",
            reason="storm response",
            created_at=START - timedelta(hours=1),
            expires_at=START + timedelta(hours=6),
        ))
        decision = board.propose(order(priority=3), crew())
        self.assertEqual(decision.decision, "reject")
        self.assertIn("override_requires_emergency:B", decision.violations)


class TestServiceIntegration(unittest.TestCase):
    def test_service_exposes_dispatch_board(self):
        svc = AquaFlowService()
        decision = svc.dispatch.propose(
            WorkOrder("WO-2", 1, frozenset({"valve"}), START, 2.0),
            [CrewMember("A", frozenset({"valve"}))],
        )
        self.assertEqual(decision.decision, "assign")
        self.assertEqual(decision.crew[0].role, "lead")


if __name__ == "__main__":
    unittest.main()
