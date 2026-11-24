"""Фаза 2. Распределение 12 задач Фазы 3 для CRM «Четверть».

Модельный учебный набор экспертных оценок: не фактическая история работы.
Выполнение: pip install numpy scipy; python phase-02-solver.py
"""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

MEMBERS = ("Александр Черноусов", "Илья Четвергов")
TASKS = (
    ("T01", "Границы MVP и роли", 2, 4, 3),
    ("T02", "Каталог 12 Use Cases", 2, 5, 3),
    ("T03", "Основные/альтернативные/негативные сценарии", 4, 5, 3),
    ("T04", "Акторы и сложность UC для UCP", 2, 2, 5),
    ("T05", "Технические и средовые факторы UCP", 2, 2, 5),
    ("T06", "Движения COSMIC", 4, 2, 5),
    ("T07", "Контроль COSMIC", 2, 2, 5),
    ("T08", "Индивидуальные оценки трудозатрат UC", 2, 3, 4),
    ("T09", "Агрегирование экспертных оценок", 2, 3, 4),
    ("T10", "Сравнение методов, сроки и команда", 3, 5, 3),
    ("T11", "Отчёт Фазы 3", 3, 5, 2),
    ("T12", "Перекрёстная проверка и приёмка", 2, 3, 5),
)
HOURS = np.array([t[2] for t in TASKS], dtype=float)
PREFERENCES = np.array([[t[3], t[4]] for t in TASKS], dtype=float)
CAPACITY = np.array([15, 15], dtype=float)
N, M = len(TASKS), len(MEMBERS)

# Каждая задача получает одного основного исполнителя; каждый — минимум две.
rows, lb, ub = [], [], []
for i in range(N):
    row = np.zeros(N * M)
    row[i * M:(i + 1) * M] = 1
    rows.append(row); lb.append(1); ub.append(1)
for j in range(M):
    row = np.zeros(N * M)
    row[j::M] = HOURS
    rows.append(row); lb.append(0); ub.append(CAPACITY[j])
for j in range(M):
    row = np.zeros(N * M)
    row[j::M] = 1
    rows.append(row); lb.append(2); ub.append(N)

# scipy.milp минимизирует, поэтому целевая функция со знаком минус.
result = milp(
    c=-PREFERENCES.flatten(),
    integrality=np.ones(N * M),
    bounds=Bounds(np.zeros(N * M), np.ones(N * M)),
    constraints=LinearConstraint(np.stack(rows), np.array(lb), np.array(ub)),
    options={"mip_rel_gap": 0.0},
)
if not result.success:
    raise RuntimeError(result.message)
X = np.rint(result.x).astype(int).reshape(N, M)
assert np.all(X.sum(axis=1) == 1)
assert np.all(X.sum(axis=0) >= 2)
assert np.all(HOURS @ X <= CAPACITY + 1e-9)
score = int(np.sum(PREFERENCES * X))

# Независимая проверка полным перебором 2^12 назначений.
possible = []
for assignment in itertools.product(range(M), repeat=N):
    load = [sum(HOURS[i] for i, j in enumerate(assignment) if j == k) for k in range(M)]
    counts = [assignment.count(k) for k in range(M)]
    if all(load[k] <= CAPACITY[k] and counts[k] >= 2 for k in range(M)):
        value = int(sum(PREFERENCES[i, j] for i, j in enumerate(assignment)))
        possible.append((value, assignment))
max_score = max(v for v, a in possible)
assert score == max_score == 53
assert sum(1 for v, a in possible if v == max_score) == 1
assert len(possible) == 392

print("CRM «Четверть» — оптимальное назначение учебных задач Фазы 3")
for i, (code, name, duration, *_rest) in enumerate(TASKS):
    assigned = int(np.argmax(X[i]))
    print(f"{code} | {MEMBERS[assigned]:19} | {duration:>2} ч | {int(PREFERENCES[i,assigned])} балл. | {name}")
for j, name in enumerate(MEMBERS):
    print(f"{name}: {int(HOURS @ X[:,j])} из {int(CAPACITY[j])} ч, {int(sum(X[:,j]))} задач")
print(f"F* = {score}; допустимых = {len(possible)}; оптимум единственный. Проверено 4096 вариантов.")
print("Исходные часы и предпочтения демонстрационные; реальные часы фиксируются в YouGile.")
