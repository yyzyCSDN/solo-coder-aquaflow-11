"""Urgent work-order crew dispatch with fatigue protection.

The dispatch board merges four signals into one assignment plan:

* work-order priority (only emergency-priority orders accept supervisor
  overrides; routine work waits for a rested crew),
* confined-space / lockout-tagout role safety requirements,
* crew qualifications (skills), and
* recent duty hours (trailing 24h / 7d limits plus minimum rest).

Every proposal persists the fatigue calculation it relied on, both on the
returned decision and in the board's audit log.  Supervisors may override a
fatigue rejection by supplying a reason and an expiry time; overrides never
modify the recorded duty history, are only honored while active at the
scheduled work start, and are always written to the audit log.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

EMERGENCY_PRIORITY = 1


@dataclass(frozen=True)
class WorkOrder:
    order_id: str
    priority: int
    required_skills: frozenset[str]
    starts: datetime
    estimated_hours: float
    confined_space: bool = False
    lockout_tagout: bool = False


@dataclass(frozen=True)
class CrewMember:
    crew_id: str
    skills: frozenset[str]


@dataclass(frozen=True)
class DutyEntry:
    crew_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class FatigueLimits:
    max_hours_24h: float = 12.0
    max_hours_7d: float = 60.0
    min_rest_hours: float = 8.0


@dataclass(frozen=True)
class FatigueReport:
    """The persisted fatigue calculation behind one dispatch decision."""

    crew_id: str
    at: datetime
    hours_24h: float
    hours_7d: float
    rest_hours: float | None
    projected_hours_24h: float
    projected_hours_7d: float
    fatigued: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Override:
    override_id: str
    order_id: str
    crew_id: str
    supervisor: str
    reason: str
    created_at: datetime
    expires_at: datetime

    def active(self, at: datetime) -> bool:
        return self.created_at <= at < self.expires_at


@dataclass(frozen=True)
class Assignment:
    crew_id: str
    role: str


@dataclass(frozen=True)
class DispatchDecision:
    order_id: str
    decision: str
    priority: int
    crew: tuple[Assignment, ...]
    fatigue: tuple[FatigueReport, ...]
    violations: tuple[str, ...]
    overrides_applied: tuple[str, ...]
    audit_seq: int


def _overlap_hours(entry: DutyEntry, lo: datetime, hi: datetime) -> float:
    span = min(entry.end, hi) - max(entry.start, lo)
    return max(0.0, span.total_seconds() / 3600.0)


class DispatchBoard:
    """Stateful dispatch board: duty history, overrides, decisions, audit."""

    def __init__(self, limits: FatigueLimits | None = None) -> None:
        self._limits = limits or FatigueLimits()
        self._duty: list[DutyEntry] = []
        self._overrides: list[Override] = []
        self._decisions: list[DispatchDecision] = []
        self._audit: list[dict[str, Any]] = []
        self._seq = 0

    @property
    def limits(self) -> FatigueLimits:
        return self._limits

    def record_duty(self, entry: DutyEntry) -> None:
        if entry.end <= entry.start:
            raise ValueError("duty end must be after start")
        self._duty.append(entry)

    def duty_history(self) -> tuple[DutyEntry, ...]:
        return tuple(self._duty)

    def decisions(self) -> tuple[DispatchDecision, ...]:
        return tuple(self._decisions)

    def audit_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._audit)

    def add_override(self, override: Override) -> None:
        if not override.reason.strip():
            raise ValueError("override requires a reason")
        if override.expires_at <= override.created_at:
            raise ValueError("override expiry must be after creation")
        self._overrides.append(override)
        self._log("override", override.created_at, {
            "override_id": override.override_id,
            "order_id": override.order_id,
            "crew_id": override.crew_id,
            "supervisor": override.supervisor,
            "reason": override.reason,
            "expires_at": override.expires_at.isoformat(),
        })

    def propose(
        self,
        order: WorkOrder,
        members: list[CrewMember],
        at: datetime | None = None,
    ) -> DispatchDecision:
        if order.estimated_hours <= 0:
            raise ValueError("estimated hours must be positive")
        decided_at = at or order.starts
        reports = tuple(self._fatigue(m, order) for m in members)
        by_id = {r.crew_id: r for r in reports}

        emergency = order.priority <= EMERGENCY_PRIORITY
        active: dict[str, list[Override]] = {}
        if emergency:
            for o in self._overrides:
                if o.order_id == order.order_id and o.active(order.starts):
                    active.setdefault(o.crew_id, []).append(o)

        best = self._select(order, members, by_id, set(active))
        violations: list[str] = []
        assignments: tuple[Assignment, ...] = ()
        applied: tuple[str, ...] = ()
        if best is None:
            violations.extend(self._structural_violations(order, members))
        else:
            crew, unfit = best
            assignments = self._roles(order, crew)
            for m in unfit:
                r = by_id[m.crew_id]
                violations.append("fatigue:%s:%s" % (m.crew_id, ",".join(r.reasons)))
                violations.extend(self._override_notes(order, m.crew_id, emergency))
            applied = tuple(sorted(
                o.override_id
                for m in crew
                if m.crew_id in active and by_id[m.crew_id].fatigued
                for o in active[m.crew_id]
            ))

        record = DispatchDecision(
            order_id=order.order_id,
            decision="reject" if violations else "assign",
            priority=order.priority,
            crew=assignments,
            fatigue=reports,
            violations=tuple(violations),
            overrides_applied=applied,
            audit_seq=self._seq + 1,
        )
        self._decisions.append(record)
        self._log("decision", decided_at, {
            "order_id": order.order_id,
            "priority": order.priority,
            "decision": record.decision,
            "crew": [(a.crew_id, a.role) for a in assignments],
            "violations": list(violations),
            "overrides_applied": list(applied),
            "fatigue": [self._fatigue_snapshot(r) for r in reports],
        })
        return record

    def _fatigue(self, member: CrewMember, order: WorkOrder) -> FatigueReport:
        start = order.starts
        day_lo = start - timedelta(hours=24)
        week_lo = start - timedelta(days=7)
        hours_24h = 0.0
        hours_7d = 0.0
        last_end: datetime | None = None
        for e in self._duty:
            if e.crew_id != member.crew_id:
                continue
            hours_24h += _overlap_hours(e, day_lo, start)
            hours_7d += _overlap_hours(e, week_lo, start)
            if e.end <= start and (last_end is None or e.end > last_end):
                last_end = e.end
        rest = None if last_end is None else (start - last_end).total_seconds() / 3600.0
        projected_24h = hours_24h + order.estimated_hours
        projected_7d = hours_7d + order.estimated_hours
        reasons: list[str] = []
        if projected_24h > self._limits.max_hours_24h:
            reasons.append("daily_hours_exceeded")
        if projected_7d > self._limits.max_hours_7d:
            reasons.append("weekly_hours_exceeded")
        if rest is not None and rest < self._limits.min_rest_hours:
            reasons.append("insufficient_rest")
        return FatigueReport(
            crew_id=member.crew_id,
            at=start,
            hours_24h=hours_24h,
            hours_7d=hours_7d,
            rest_hours=rest,
            projected_hours_24h=projected_24h,
            projected_hours_7d=projected_7d,
            fatigued=bool(reasons),
            reasons=tuple(reasons),
        )

    def _select(
        self,
        order: WorkOrder,
        members: list[CrewMember],
        reports: dict[str, FatigueReport],
        overridden: set[str],
    ) -> tuple[tuple[CrewMember, ...], tuple[CrewMember, ...]] | None:
        best = None
        for mask in range(1, 1 << len(members)):
            crew = tuple(members[i] for i in range(len(members)) if mask & (1 << i))
            skills = set().union(*(m.skills for m in crew))
            if not set(order.required_skills) <= skills:
                continue
            if order.confined_space and len(crew) < 3:
                continue
            if order.lockout_tagout and not any("lockout" in m.skills for m in crew):
                continue
            unfit = tuple(
                m for m in crew
                if reports[m.crew_id].fatigued and m.crew_id not in overridden
            )
            key = (
                len(unfit),
                len(crew),
                round(sum(reports[m.crew_id].projected_hours_24h for m in crew), 6),
                tuple(sorted(m.crew_id for m in crew)),
            )
            if best is None or key < best[0]:
                best = (key, crew, unfit)
        return None if best is None else (best[1], best[2])

    @staticmethod
    def _roles(order: WorkOrder, crew: tuple[CrewMember, ...]) -> tuple[Assignment, ...]:
        ordered = sorted(crew, key=lambda m: m.crew_id)
        if order.confined_space:
            labels = ("entrant", "attendant", "rescue")
            return tuple(
                Assignment(m.crew_id, labels[i] if i < len(labels) else "crew")
                for i, m in enumerate(ordered)
            )
        return tuple(
            Assignment(m.crew_id, "lead" if i == 0 else "crew")
            for i, m in enumerate(ordered)
        )

    @staticmethod
    def _structural_violations(order: WorkOrder, members: list[CrewMember]) -> list[str]:
        out: list[str] = []
        skills = set().union(*(m.skills for m in members)) if members else set()
        missing = sorted(set(order.required_skills) - skills)
        if missing:
            out.append("missing_skills:" + ",".join(missing))
        if order.confined_space and len(members) < 3:
            out.append("confined_space_requires_3_distinct_crew")
        if order.lockout_tagout and not any("lockout" in m.skills for m in members):
            out.append("lockout_qualified_crew_required")
        if not out:
            out.append("no_feasible_crew")
        return out

    def _override_notes(self, order: WorkOrder, crew_id: str, emergency: bool) -> list[str]:
        notes: list[str] = []
        for o in self._overrides:
            if o.order_id != order.order_id or o.crew_id != crew_id:
                continue
            if o.active(order.starts):
                if not emergency:
                    notes.append(f"override_requires_emergency:{crew_id}")
            elif order.starts >= o.expires_at:
                notes.append(f"override_expired:{crew_id}")
            else:
                notes.append(f"override_not_yet_active:{crew_id}")
        return notes

    @staticmethod
    def _fatigue_snapshot(r: FatigueReport) -> dict[str, Any]:
        return {
            "crew_id": r.crew_id,
            "at": r.at.isoformat(),
            "hours_24h": r.hours_24h,
            "hours_7d": r.hours_7d,
            "rest_hours": r.rest_hours,
            "projected_hours_24h": r.projected_hours_24h,
            "projected_hours_7d": r.projected_hours_7d,
            "fatigued": r.fatigued,
            "reasons": list(r.reasons),
        }

    def _log(self, kind: str, at: datetime, detail: dict[str, Any]) -> None:
        self._seq += 1
        self._audit.append({"seq": self._seq, "kind": kind, "at": at.isoformat(), **detail})
