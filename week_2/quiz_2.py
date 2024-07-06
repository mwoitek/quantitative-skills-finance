# %%
from typing import cast

from sympy import Eq, Float, Pow, Symbol, latex, simplify
from sympy.sets.sets import Set
from sympy.solvers.solveset import solveset_real


# %%
def npv(
    initial_investment: float,
    cash_flow: float | list[float],
    discount_rate: float,
    time_periods: int,
) -> float:
    if not isinstance(cash_flow, list):
        cash_flow = [float(cash_flow)] * time_periods
    cash_flow = cast(list, cash_flow)
    pv_cash_flow = sum(f / (1 + discount_rate) ** t for f, t in zip(cash_flow, range(1, time_periods + 1)))
    return pv_cash_flow - initial_investment


# %%
discount_rate = 0.05
time_periods = 3

# %%
initial_investment = 1000
cash_flow = 500

ans_1a = npv(initial_investment, cash_flow, discount_rate, time_periods)
print(round(ans_1a, 2))

# %%
initial_investment = 2000
cash_flow = 900

ans_1b = npv(initial_investment, cash_flow, discount_rate, time_periods)
print(round(ans_1b, 2))


# %%
def irr_equation(initial_investment: int | float, cash_flow: list[int] | list[float]) -> tuple[Eq, Set]:
    irr = Symbol("IRR")
    npv = Float(-initial_investment)
    for t, f in enumerate(cash_flow, start=1):
        npv += Float(f) / Pow(Float(1) + irr, t)
    lhs = npv * Pow(Float(1) + irr, len(cash_flow))
    equation = simplify(Eq(lhs, 0))
    solutions = solveset_real(equation, irr)
    return equation, solutions


# %%
initial_investment = 1000
cash_flow = [50, 1050]
eq_1, sol_1 = irr_equation(initial_investment, cash_flow)
print(latex(eq_1))
print(sol_1)

# %%
cash_flow = [40, 1040]
eq_2, sol_2 = irr_equation(initial_investment, cash_flow)
print(latex(eq_2))
print(sol_2)

# %%
initial_investment = 1050
cash_flow = [50, 1050]
eq_3, sol_3 = irr_equation(initial_investment, cash_flow)
print(latex(eq_3))
print(sol_3)
