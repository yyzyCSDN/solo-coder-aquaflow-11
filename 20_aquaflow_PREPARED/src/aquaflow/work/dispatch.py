"""Fatigue-protected dispatch for urgent field work.

The dispatcher combines four independent safety inputs:

* work-order priority and start time;
* confined-space and lockout/tagout requirements;
* crew qualifications and explicit role assignments;
* duty hours worked during the look-back window ending at work start.

Supervisor overrides may permit a fatigued crew, but they never alter role
rules, qualifications, or the duty history used to calculate fatigue.  Every
plan and override decision is represented in the immutable audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Iterable, Mapping


ENTRANT = "entrant"
ATTENDANT = "attendant"
RESCUE = "rescue"
LOTO_AUTHORIZED = "loto_authorized"
CREW_LEAD = "crew_lead"

CONFIDENT_ROLES = frozenset({ENTRANT, ATTENDANT, RESCUE})

ROLE_ALIASES: dict[str, str] = {
    "crew_lead": CREW_LEAD,
    "lead": CREW_LEAD,
    "supervisor": CREW_LEAD,
    "incident_commander": CREW_LEAD,
    "emergency_lead": CREW_LEAD,
    ENTRANT: ENTRANT,
    "confined_entrant": ENTRANT,
    "confined_space_entrant": ENTRANT,
    ATTENDANT: ATTENDANT,
    "confined_attendant": ATTENDANT,
    "confined_space_attendant": ATTENDANT,
    "attendant": ATTENDANT,
    RESCUE: RESCUE,
    "confined_rescue": RESCUE,
    "confined_space_rescue": RESCUE,
    "rescue": RESCUE,
    LOTO_AUTHORIZED: LOTO_AUTHORIZED,
    "authorized_employee": LOTO_AUTHORIZED,
    "loto_authorized_employee": LOTO_AUTHORIZED,
    "lockout_authorized": LOTO_AUTHORIZED,
    "lockout_tagout_authorized": LOTO_AUTHORIZED,
}

ROLE_QUALIFICATIONS: dict[str, frozenset[str]] = {
    CREW_LEAD: frozenset({"crew_lead", "lead", "supervisor", "incident_commander", "emergency_lead"}),
    ENTRANT: frozenset({"entrant", "confined_entrant", "confined_space_entrant", "confined_space", "confined_space_certified"}),
    ATTENDANT: frozenset({"attendant", "confined_attendant", "confined_space_attendant", "confined_space", "confined_space_certified"}),
    RESCUE: frozenset({"rescue", "confined_rescue", "confined_space_rescue", "confined_space", "confined_space_certified"}),
    LOTO_AUTHORIZED: frozenset({"loto_authorized", "authorized_employee", "loto", "lockout_tagout", "lockout_tagout_authorized"}),
}


@dataclass(frozen=True)
class WorkOrder:
    id: str
    priority: int
    starts_at: datetime | None = None
    confined_space: bool = False
    lockout_tagout: bool = False
    required_qualifications: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_qualifications", _qualification_set(self.required_qualifications))


@dataclass(frozen=True)
class Worker:
    id: str
    qualifications: frozenset[str] = field(default_factory=frozenset)
    recent_duty_hours: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.qualifications, frozenset):
            object.__setattr__(self, "qualifications", _qualification_set(self.qualifications))
        else:
            object.__setattr__(self, "qualifications", _qualification_set(self.qualifications))


@dataclass(frozen=True)
class Crew:
    id: str
    members: tuple[Worker, ...]
    role_assignments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DutyShift:
    worker_id: str
    starts_at: datetime
    ends_at: datetime


# A concise name for callers that represent duty as hour totals rather than
# individual shift intervals.
DutyRecord = DutyShift
Shift = DutyShift


@dataclass(frozen=True)
class FatiguePolicy:
    limit_hours: float = 12.0
    lookback_hours: float = 24.0
    emergency_limit_hours: float = 12.0

    def limit_for_priority(self, priority: int) -> float:
        if priority < 1 or priority > 5:
            raise ValueError("priority outside 1..5")
        return self.emergency_limit_hours if priority <= 2 else self.limit_hours


@dataclass(frozen=True)
class FatigueCalculation:
    worker_id: str
    window_start: datetime
    window_end: datetime
    duty_hours: float
    limit_hours: float
    fatigued: bool
    intervals: tuple[DutyShift, ...] = ()


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    blocking: bool = True
    worker_id: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class SupervisorOverride:
    id: str
    work_order_id: str
    crew_id: str
    supervisor: str
    reason: str
    created_at: datetime
    expires_at: datetime
    permissions: frozenset[str] = field(default_factory=lambda: frozenset({"fatigue"}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "permissions", _qualification_set(self.permissions))


@dataclass(frozen=True)
class CrewEvaluation:
    crew_id: str
    allowed: bool
    role_assignments: tuple[tuple[str, str], ...]
    violations: tuple[Violation, ...]
    fatigue: tuple[FatigueCalculation, ...]
    override: SupervisorOverride | None = None
    override_applied: bool = False
    override_active: bool | None = None

    @property
    def max_duty_hours(self) -> float:
        return max((row.duty_hours for row in self.fatigue), default=0.0)


@dataclass(frozen=True)
class CrewAssignment:
    crew_id: str
    workers: tuple[str, ...]
    role_assignments: tuple[tuple[str, str], ...]
    fatigue: tuple[FatigueCalculation, ...]
    override: SupervisorOverride | None = None


@dataclass(frozen=True)
class DispatchPlan:
    work_order_id: str
    priority: int
    starts_at: datetime
    decision: str
    selected: CrewAssignment | None
    options: tuple[CrewEvaluation, ...]
    fatigue_calculations: tuple[FatigueCalculation, ...]
    override: SupervisorOverride | None = None
    override_applied: bool = False

    @property
    def rejected(self) -> tuple[CrewEvaluation, ...]:
        return tuple(option for option in self.options if not option.allowed)

    @property
    def denials(self) -> tuple[CrewEvaluation, ...]:
        return self.rejected


@dataclass(frozen=True)
class AuditEntry:
    id: int
    action: str
    at: datetime
    decision: str
    work_order_id: str | None = None
    crew_id: str | None = None
    supervisor: str | None = None
    reason: str | None = None
    override_id: str | None = None
    expires_at: datetime | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "action": self.action,
            "at": self.at.isoformat(),
            "decision": self.decision,
            "work_order_id": self.work_order_id,
            "crew_id": self.crew_id,
        }
        if self.supervisor is not None:
            out["supervisor"] = self.supervisor
        if self.reason is not None:
            out["reason"] = self.reason
        if self.override_id is not None:
            out["override_id"] = self.override_id
        if self.expires_at is not None:
            out["expires_at"] = self.expires_at.isoformat()
        if self.details:
            out["details"] = dict(self.details)
        return out


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _strings(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)


def _qualification_set(value: Any) -> frozenset[str]:
    return frozenset(str(item).strip().lower().replace("-", "_").replace(" ", "_") for item in _strings(value))


def _bool(value: Any, *names: str, source: Mapping[str, Any]) -> bool:
    for name in names:
        if name in source:
            return bool(source[name])
    return bool(value)


def as_work_order(value: Any) -> WorkOrder:
    if isinstance(value, WorkOrder):
        return value
    if is_dataclass(value):
        return WorkOrder(
            str(getattr(value, "id")),
            int(getattr(value, "priority")),
            getattr(value, "starts_at", None),
            bool(getattr(value, "confined_space", False)),
            bool(getattr(value, "lockout_tagout", False)),
            _qualification_set(getattr(value, "required_qualifications", frozenset())),
        )
    if not isinstance(value, Mapping):
        raise TypeError("work order required")
    starts_at = value.get("starts_at", value.get("start_at", value.get("starts", value.get("start"))))
    return WorkOrder(
        str(value.get("id", value.get("work_order_id"))),
        int(value["priority"]),
        _dt(starts_at) if starts_at is not None else None,
        _bool(False, "confined_space", "requires_confined_space", "confined", source=value),
        _bool(False, "lockout_tagout", "requires_lockout_tagout", "loto", source=value),
        _qualification_set(value.get("required_qualifications", value.get("qualifications", frozenset()))),
    )


def as_worker(value: Any) -> Worker:
    if isinstance(value, Worker):
        return value
    if is_dataclass(value):
        return Worker(
            str(getattr(value, "id")),
            _qualification_set(getattr(value, "qualifications", frozenset())),
            getattr(value, "recent_duty_hours", None),
        )
    if isinstance(value, str):
        return Worker(value)
    if not isinstance(value, Mapping):
        raise TypeError("worker required")
    qualifications = value.get("qualifications", value.get("certifications", value.get("tickets", frozenset())))
    hours = value.get("recent_duty_hours", value.get("duty_hours_last_24", value.get("hours_last_24")))
    return Worker(str(value.get("id", value.get("worker_id"))), _qualification_set(qualifications), None if hours is None else float(hours))


def _canonical_role(role: Any) -> str | None:
    return ROLE_ALIASES.get(str(role)) or ROLE_ALIASES.get(str(role).strip().lower().replace("-", "_").replace(" ", "_"))


def as_crew(value: Any) -> Crew:
    if isinstance(value, Crew):
        return value
    if is_dataclass(value):
        members = tuple(as_worker(item) for item in getattr(value, "members"))
        assignments = tuple(getattr(value, "role_assignments", ()))
        crew_id = str(getattr(value, "id"))
    elif isinstance(value, Mapping):
        crew_id = str(value.get("id", value.get("crew_id")))
        members = tuple(as_worker(item) for item in value.get("members", value.get("workers", ())))
        raw_assignments = value.get("role_assignments", value.get("assignments", ()))
        if isinstance(raw_assignments, Mapping):
            assignments = tuple(raw_assignments.items())
        else:
            assignments = tuple(raw_assignments)
    else:
        raise TypeError("crew required")

    normalized: list[tuple[str, str]] = []
    for item in assignments:
        role, worker_id = item
        canonical = _canonical_role(role)
        normalized.append((canonical or str(role), str(worker_id)))
    return Crew(crew_id, members, tuple(normalized))


def as_shift(value: Any) -> DutyShift:
    if isinstance(value, DutyShift):
        return value
    if is_dataclass(value):
        return DutyShift(
            str(getattr(value, "worker_id")),
            _dt(getattr(value, "starts_at")),
            _dt(getattr(value, "ends_at")),
        )
    if not isinstance(value, Mapping):
        raise TypeError("duty shift required")
    return DutyShift(
        str(value.get("worker_id", value.get("worker"))),
        _dt(value.get("starts_at", value.get("start_at", value.get("start")))),
        _dt(value.get("ends_at", value.get("end_at", value.get("end")))),
    )


def as_override(value: Any) -> SupervisorOverride:
    if isinstance(value, SupervisorOverride):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("supervisor override required")
    permissions = value.get("permissions", ("fatigue",))
    return SupervisorOverride(
        str(value.get("id", value.get("override_id", "override"))),
        str(value["work_order_id"]),
        str(value["crew_id"]),
        str(value["supervisor"]),
        str(value["reason"]),
        _dt(value["created_at"]),
        _dt(value["expires_at"]),
        _qualification_set(permissions),
    )


def required_roles(order: WorkOrder) -> tuple[str, ...]:
    roles = [CREW_LEAD] if order.priority <= 2 else []
    if order.confined_space:
        roles.extend([ENTRANT, ATTENDANT, RESCUE])
    if order.lockout_tagout:
        roles.append(LOTO_AUTHORIZED)
    return tuple(roles)


def _qualified_for_role(worker: Worker, role: str) -> bool:
    return bool(worker.qualifications & ROLE_QUALIFICATIONS[role])


def _role_violations(order: WorkOrder, crew: Crew) -> tuple[tuple[str, str], tuple[Violation, ...]]:
    roles = required_roles(order)
    members = {worker.id: worker for worker in crew.members}
    violations: list[Violation] = []
    explicit_assignments = dict(crew.role_assignments)
    assignments: dict[str, str] = {}

    for role, worker_id in explicit_assignments.items():
        if role not in roles:
            violations.append(Violation("unknown_role_assignment", f"{role} is not required on this work order", False, worker_id, role))
        elif worker_id not in members:
            violations.append(Violation("worker_not_in_crew", f"{worker_id} is not assigned to crew {crew.id}", True, worker_id, role))
        elif not _qualified_for_role(members[worker_id], role):
            violations.append(Violation("missing_role_qualification", f"{worker_id} is not qualified as {role}", True, worker_id, role))

    assignments.update(explicit_assignments)
    if explicit_assignments:
        for role in roles:
            if role in assignments:
                continue
            qualified = [worker.id for worker in crew.members if _qualified_for_role(worker, role)]
            if qualified:
                violations.append(Violation("missing_role_assignment", f"no worker assigned as {role}", True, role=role))
                missing_explicit_roles = True
            else:
                violations.append(Violation("missing_role_qualification", f"crew has no qualified {role}", True, role=role))

    confined_workers = [assignments[role] for role in CONFIDENT_ROLES if role in assignments]
    if len(confined_workers) != len(set(confined_workers)):
        violations.append(Violation(
            "unsafe_role_combination",
            "confined-space entrant, attendant, and rescue must be different workers",
            True,
        ))

    if violations:
        return tuple((role, assignments[role]) for role in roles if role in assignments), tuple(violations)

    choices: dict[str, tuple[str, ...]] = {}
    for role in roles:
        if role in assignments:
            choices[role] = (assignments[role],)
            continue
        qualified = tuple(worker.id for worker in crew.members if _qualified_for_role(worker, role))
        if not qualified:
            return (), (Violation("missing_role_qualification", f"crew has no qualified {role}", True, role=role),)
        choices[role] = qualified

    distinct_roles = tuple(role for role in roles if role in CONFIDENT_ROLES)
    shared_roles = tuple(role for role in roles if role not in CONFIDENT_ROLES)
    for distinct_values in product(*(choices[role] for role in distinct_roles)):
        if len(set(distinct_values)) != len(distinct_values):
            continue
        selected = dict(zip(distinct_roles, distinct_values))
        for role in shared_roles:
            candidates = choices[role]
            # LOTO can overlap with an already-selected person; other shared
            # roles deterministically use the first qualified candidate.
            if role == LOTO_AUTHORIZED:
                selected[role] = min(candidates, key=lambda worker_id: (worker_id not in selected.values(), worker_id))
            else:
                selected[role] = candidates[0]
        return tuple((role, selected[role]) for role in roles), ()

    return (), (Violation("unsafe_role_combination", "crew cannot fill required roles without combining incompatible duties", True),)


def _mapping_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("worker_id", row.get("worker", row.get("id")))
    return None if value is None else str(value)


def _mapping_hours(row: Any) -> float | None:
    if isinstance(row, Mapping):
        for key in ("hours", "duty_hours", "recent_duty_hours", "hours_last_24"):
            if key in row:
                return float(row[key])
    return None


def _matching_shifts(worker_id: str, history: Any) -> tuple[DutyShift, ...]:
    if history is None:
        return ()
    if isinstance(history, Mapping):
        value = history.get(worker_id, ())
        if isinstance(value, (int, float)):
            return ()
        if isinstance(value, DutyShift):
            return (value,)
        return tuple(
            as_shift(item)
            for item in value
            if isinstance(item, DutyShift)
            or (isinstance(item, Mapping) and any(key in item for key in ("starts_at", "start_at", "starts", "start")))
        )
    return tuple(
        as_shift(item)
        for item in history
        if (getattr(item, "worker_id", None) == worker_id and hasattr(item, "starts_at"))
        or (isinstance(item, Mapping)
            and _mapping_id(item) == worker_id
            and any(key in item for key in ("starts_at", "start_at", "starts", "start")))
    )


def _matching_hours(worker_id: str, history: Any) -> float | None:
    if history is None:
        return None
    total: float | None = None

    def add(value: Any) -> None:
        nonlocal total
        if isinstance(value, (int, float)):
            total = float(value) if total is None else total + float(value)
        elif isinstance(value, Mapping):
            hours = _mapping_hours(value)
            if hours is not None:
                total = hours if total is None else total + hours
        else:
            for row in value:
                if isinstance(row, DutyShift):
                    continue
                if isinstance(row, Mapping) and any(key in row for key in ("starts_at", "start_at", "starts", "start")):
                    continue
                if isinstance(row, Mapping) and _mapping_id(row) == worker_id:
                    add(row)

    if isinstance(history, Mapping):
        add(history.get(worker_id))
    else:
        for row in history:
            if getattr(row, "worker_id", None) == worker_id and not hasattr(row, "starts_at"):
                add(getattr(row, "hours", getattr(row, "duty_hours", None)))
            elif isinstance(row, Mapping) and _mapping_id(row) == worker_id:
                add(row)
    return total


def _direct_hours(worker: Worker, history: Any, matched_shift: bool, at: datetime) -> float | None:
    hours = None if matched_shift else _matching_hours(worker.id, history)
    if hours is not None:
        return hours
    if not matched_shift and worker.recent_duty_hours is not None:
        return float(worker.recent_duty_hours)
    return None


def fatigue_for_worker(
    worker: Worker,
    at: datetime,
    policy: FatiguePolicy | None = None,
    priority: int = 3,
    history: Any = None,
) -> FatigueCalculation:
    policy = policy or FatiguePolicy()
    window_end = at
    window_start = at - timedelta(hours=policy.lookback_hours)
    shifts = _matching_shifts(worker.id, history)
    used_intervals: list[DutyShift] = []
    seconds = 0.0

    for shift in shifts:
        if shift.ends_at < shift.starts_at:
            raise ValueError(f"duty shift for {worker.id} ends before it starts")
        overlap_start = max(shift.starts_at, window_start)
        overlap_end = min(shift.ends_at, window_end)
        if overlap_start < overlap_end:
            seconds += (overlap_end - overlap_start).total_seconds()
            used_intervals.append(shift)

    direct_hours = _direct_hours(worker, history, bool(used_intervals), at)
    if direct_hours is not None:
        if direct_hours < 0:
            raise ValueError("negative duty hours")
        duty_hours = direct_hours
    else:
        duty_hours = seconds / 3600.0

    limit = policy.limit_for_priority(priority)
    return FatigueCalculation(
        worker.id,
        window_start,
        window_end,
        duty_hours,
        limit,
        duty_hours > limit,
        tuple(used_intervals),
    )


def _crew_qualification_violations(order: WorkOrder, crew: Crew) -> tuple[Violation, ...]:
    available: frozenset[str] = frozenset().union(*(worker.qualifications for worker in crew.members))
    missing = tuple(sorted(order.required_qualifications - available))
    return tuple(
        Violation("missing_required_qualification", f"crew lacks qualification {qualification}", True, role=qualification)
        for qualification in missing
    )


def override_active_at(override: SupervisorOverride, at: datetime) -> bool:
    return override.expires_at > at


def evaluate_crew(
    order: WorkOrder | Mapping[str, Any],
    crew: Crew | Mapping[str, Any],
    history: Any = (),
    at: datetime | None = None,
    override: SupervisorOverride | Mapping[str, Any] | None = None,
    policy: FatiguePolicy | None = None,
) -> CrewEvaluation:
    order = as_work_order(order)
    crew = as_crew(crew)
    if override is not None:
        override = as_override(override)
    starts_at = order.starts_at or at or datetime.now().astimezone()
    policy = policy or FatiguePolicy()

    violations = list(_crew_qualification_violations(order, crew))
    assignments, role_violations = _role_violations(order, crew)
    violations.extend(role_violations)

    fatigue = tuple(fatigue_for_worker(worker, starts_at, policy, order.priority, history) for worker in crew.members)
    fatigued_workers = [row.worker_id for row in fatigue if row.fatigued]
    for worker_id in fatigued_workers:
        violations.append(Violation("fatigued_worker", f"{worker_id} exceeds the recent duty-hour limit", False, worker_id))

    safety_blocked = any(violation.blocking for violation in violations if violation.code != "fatigued_worker")
    matching_override = override and override.work_order_id == order.id and override.crew_id == crew.id
    override_active = None
    override_applied = False
    if override is not None:
        if not matching_override:
            override_active = False
            violations.append(Violation("override_scope_mismatch", "override is for a different work order or crew", False))
        elif safety_blocked:
            override_active = override_active_at(override, starts_at)
            violations.append(Violation("override_not_permitted", "supervisor override cannot bypass unsafe role or qualification rules", True))
        else:
            override_active = override_active_at(override, starts_at)
            if fatigued_workers and "fatigue" in override.permissions and override_active:
                override_applied = True
            elif fatigued_workers and not override_active:
                violations.append(Violation("expired_override", "supervisor override expired before work start", True))
            elif fatigued_workers:
                violations.append(Violation("override_not_permitted", "override does not permit fatigue", True))

    hard_blocked = any(violation.blocking for violation in violations if violation.code != "fatigued_worker")
    fatigue_blocked = bool(fatigued_workers) and not override_applied
    return CrewEvaluation(
        crew.id,
        not hard_blocked and not fatigue_blocked,
        assignments,
        tuple(violations),
        fatigue,
        override if matching_override else None,
        override_applied,
        override_active,
    )


class DispatchService:
    """Stateful dispatcher that retains fatigue evidence and audit records."""

    def __init__(self, policy: FatiguePolicy | None = None):
        self.policy = policy or FatiguePolicy()
        self.fatigue_records: dict[tuple[str, str, str, datetime], FatigueCalculation] = {}
        self._audit: list[AuditEntry] = []

    @property
    def audit_trail(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit)

    def audit_log(self) -> list[dict[str, Any]]:
        return [entry.as_dict() for entry in self._audit]

    def audit_records(self) -> tuple[AuditEntry, ...]:
        return self.audit_trail

    def _record_audit(self, action: str, at: datetime, decision: str, **kwargs: Any) -> AuditEntry:
        entry = AuditEntry(len(self._audit) + 1, action, at, decision, **kwargs)
        self._audit.append(entry)
        return entry

    def grant_override(
        self,
        work_order_id: str,
        crew_id: str,
        supervisor: str,
        reason: str,
        created_at: datetime,
        expires_at: datetime,
        override_id: str | None = None,
    ) -> SupervisorOverride:
        created_at = _dt(created_at)
        expires_at = _dt(expires_at)
        reason = str(reason).strip()
        supervisor = str(supervisor).strip()
        if not reason:
            raise ValueError("override reason required")
        if not supervisor:
            raise ValueError("supervisor required")
        if expires_at <= created_at:
            raise ValueError("override must expire after it is created")
        override = SupervisorOverride(
            override_id or f"override-{len(self._audit) + 1}",
            str(work_order_id),
            str(crew_id),
            supervisor,
            reason,
            created_at,
            expires_at,
        )
        self._record_audit(
            "supervisor_override_granted",
            created_at,
            "recorded",
            work_order_id=override.work_order_id,
            crew_id=override.crew_id,
            supervisor=supervisor,
            reason=reason,
            override_id=override.id,
            expires_at=expires_at,
        )
        return override

    def _save_fatigue(self, order_id: str, evaluation: CrewEvaluation) -> None:
        for row in evaluation.fatigue:
            self.fatigue_records[(order_id, evaluation.crew_id, row.worker_id, row.window_end)] = row

    def _plan(
        self,
        order: WorkOrder | Mapping[str, Any],
        crews: Iterable[Any],
        history: Any = (),
        at: datetime | None = None,
        override: SupervisorOverride | Mapping[str, Any] | None = None,
        action: str = "dispatch_plan_created",
    ) -> DispatchPlan:
        order = as_work_order(order)
        if override is not None:
            override = as_override(override)
        starts_at = _dt(at) if at is not None else order.starts_at or datetime.now().astimezone()
        normalized_order = WorkOrder(
            order.id,
            order.priority,
            starts_at,
            order.confined_space,
            order.lockout_tagout,
            order.required_qualifications,
        )

        normalized_crews = [as_crew(item) for item in crews]
        evaluations = []
        for crew in normalized_crews:
            evaluation = evaluate_crew(normalized_order, crew, history, starts_at, override, self.policy)
            self._save_fatigue(order.id, evaluation)
            evaluations.append(evaluation)
            self._record_crew_evaluation(starts_at, order.id, evaluation)

        eligible = [item for item in evaluations if item.allowed]
        selected_evaluation = min(eligible, key=lambda item: (item.max_duty_hours, item.crew_id)) if eligible else None
        selected = None
        if selected_evaluation is not None:
            workers = tuple(row.worker_id for row in selected_evaluation.fatigue)
            selected = CrewAssignment(
                selected_evaluation.crew_id,
                workers,
                selected_evaluation.role_assignments,
                selected_evaluation.fatigue,
                selected_evaluation.override,
            )

        decision = "dispatch" if selected is not None else "do_not_dispatch"
        applied_override = selected_evaluation.override if selected_evaluation is not None and selected_evaluation.override_applied else None
        plan = DispatchPlan(
            order.id,
            order.priority,
            starts_at,
            decision,
            selected,
            tuple(evaluations),
            tuple(row for evaluation in evaluations for row in evaluation.fatigue),
            applied_override,
            applied_override is not None,
        )

        details = [
            ("priority", order.priority),
            ("confined_space", order.confined_space),
            ("lockout_tagout", order.lockout_tagout),
            ("rejected_crew_ids", tuple(item.crew_id for item in plan.rejected)),
        ]
        if selected is not None:
            details.append(("selected_crew_id", selected.crew_id))
        if override is not None:
            details.extend([
                ("override_id", override.id),
                ("override_reason", override.reason),
                ("override_expires_at", override.expires_at.isoformat()),
                ("override_applied", plan.override_applied),
            ])
        self._record_audit(
            action,
            starts_at,
            decision,
            work_order_id=order.id,
            crew_id=selected.crew_id if selected else None,
            supervisor=applied_override.supervisor if applied_override else (override.supervisor if override else None),
            reason=applied_override.reason if applied_override else (override.reason if override else None),
            override_id=override.id if override else None,
            expires_at=override.expires_at if override else None,
            details=tuple(details),
        )
        return plan

    def _record_crew_evaluation(self, at: datetime, order_id: str, evaluation: CrewEvaluation) -> None:
        action = "crew_evaluated"
        decision = "allowed" if evaluation.allowed else "denied"
        codes = {item.code for item in evaluation.violations}
        if evaluation.override_applied:
            action = "supervisor_override_applied"
        elif "expired_override" in codes or (evaluation.override_active is False and evaluation.override is not None):
            action = "supervisor_override_expired"
            decision = "override_inactive"
        elif "override_not_permitted" in codes:
            action = "supervisor_override_denied"
            decision = "unsafe_override_rejected"
        details = [
            ("violations", tuple(item.code for item in evaluation.violations)),
            ("max_duty_hours", evaluation.max_duty_hours),
            ("role_assignments", evaluation.role_assignments),
        ]
        self._record_audit(
            action,
            at,
            decision,
            work_order_id=order_id,
            crew_id=evaluation.crew_id,
            supervisor=evaluation.override.supervisor if evaluation.override else None,
            reason=evaluation.override.reason if evaluation.override else None,
            override_id=evaluation.override.id if evaluation.override else None,
            expires_at=evaluation.override.expires_at if evaluation.override else None,
            details=tuple(details),
        )

    def propose(
        self,
        order: WorkOrder | Mapping[str, Any],
        crews: Iterable[Any],
        history: Any = (),
        at: datetime | None = None,
        override: SupervisorOverride | Mapping[str, Any] | None = None,
    ) -> DispatchPlan:
        return self._plan(order, crews, history, at, override, action="dispatch_proposed")

    create_override = grant_override
    override = grant_override

    def dispatch(
        self,
        order: WorkOrder | Mapping[str, Any],
        crew: Any,
        history: Any = (),
        at: datetime | None = None,
        override: SupervisorOverride | Mapping[str, Any] | None = None,
    ) -> DispatchPlan:
        normalized_crew = as_crew(crew)
        return self._plan(order, [normalized_crew], history, at, override, action="dispatch_decided")


# Convenience functional API used by stateless callers and the workflow layer.
def propose_dispatch(
    order: WorkOrder | Mapping[str, Any],
    crews: Iterable[Any],
    history: Any = (),
    at: datetime | None = None,
    override: SupervisorOverride | Mapping[str, Any] | None = None,
    policy: FatiguePolicy | None = None,
) -> DispatchPlan:
    return DispatchService(policy).propose(order, crews, history, at, override)


def serialize_plan(plan: DispatchPlan) -> dict[str, Any]:
    """Return a JSON-ready representation suitable for workflow responses."""
    return {
        "work_order_id": plan.work_order_id,
        "priority": plan.priority,
        "starts_at": plan.starts_at.isoformat(),
        "decision": plan.decision,
        "selected": None
        if plan.selected is None
        else {
            "crew_id": plan.selected.crew_id,
            "workers": list(plan.selected.workers),
            "role_assignments": [[role, worker_id] for role, worker_id in plan.selected.role_assignments],
            "override_id": plan.selected.override.id if plan.selected.override else None,
        },
        "options": [
            {
                "crew_id": option.crew_id,
                "allowed": option.allowed,
                "role_assignments": [[role, worker_id] for role, worker_id in option.role_assignments],
                "violations": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "blocking": item.blocking,
                        "worker_id": item.worker_id,
                        "role": item.role,
                    }
                    for item in option.violations
                ],
                "max_duty_hours": option.max_duty_hours,
                "override_applied": option.override_applied,
                "override_active": option.override_active,
            }
            for option in plan.options
        ],
        "fatigue_calculations": [
            {
                "worker_id": row.worker_id,
                "window_start": row.window_start.isoformat(),
                "window_end": row.window_end.isoformat(),
                "duty_hours": row.duty_hours,
                "limit_hours": row.limit_hours,
                "fatigued": row.fatigued,
            }
            for row in plan.fatigue_calculations
        ],
        "override_applied": plan.override_applied,
        "override_id": plan.override.id if plan.override else None,
    }


# Backward-friendly domain aliases.
WorkOrderDispatch = WorkOrder
FatigueRecord = FatigueCalculation
SupervisorOverrideRecord = SupervisorOverride
