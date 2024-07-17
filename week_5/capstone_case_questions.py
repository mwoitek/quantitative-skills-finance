# %%
from itertools import pairwise
from pprint import pprint
from typing import cast

from sympy import Eq, Float, Pow, Symbol, latex, simplify
from sympy.sets.sets import Set
from sympy.solvers.solveset import solveset_real


# %%
def format_as_money(number: int) -> str:
    return f"-${abs(number):,.0f}" if number < 0 else f"${number:,.0f}"


# %%
# Non-cash current assets and current liabilities
curr_assets = [15000, 17025, 17175, 17325, 17475, 17475, 17475]
curr_liabilities = [0, 2700, 2900, 3100, 3300, 3300, 3300]

# %%
# Compute working capital
working_capital = [ca - cl for ca, cl in zip(curr_assets, curr_liabilities)]
# pprint([format_as_money(n) for n in working_capital])

# %%
# Compute changes in working capital
changes_wc = [working_capital[0]]
changes_wc.extend([new_wc - old_wc for old_wc, new_wc in pairwise(working_capital[:-1])])
changes_wc.append(-working_capital[-1])
# pprint([format_as_money(n) for n in changes_wc])

# %%
# Generate the table that shows the answer to Problem 7
row_1 = [""]
row_1.extend([str(i) for i in range(7)])

row_2 = ["Other Current Assets (Inventory/Receivables)"]
row_2.extend([format_as_money(n) for n in curr_assets])

row_3 = ["Current Liabilities (Payables)"]
row_3.extend([format_as_money(n) for n in curr_liabilities])

row_4 = ["Working Capital"]
row_4.extend([format_as_money(n) for n in working_capital])

row_5 = ["Change in Working Capital"]
row_5.extend([format_as_money(n) for n in changes_wc])

ans_7 = [
    row_1,
    row_2,
    row_3,
    row_4,
    row_5,
]
pprint(ans_7, width=106)

# %%
# Data we need to compute the free cash flows
capital_expenditure = 350000
net_income = [0, 56000, 61600, 67200, 72800, 72800, 72800]
depreciation = [0, 35000, 35000, 35000, 35000, 35000, 35000]
salvage_value = 140000

# %%
# Computing free cash flows
fcfs = []
for i in range(7):
    fcf = net_income[i]
    fcf += depreciation[i]
    fcf -= changes_wc[i]
    fcfs.append(fcf)
fcfs[0] -= capital_expenditure
fcfs[-1] += salvage_value
# pprint([format_as_money(n) for n in fcfs], width=106)

# %%
# Generate the table that shows the answer to Problem 9
ans_9 = [
    [str(i) for i in range(7)],
    [format_as_money(n) for n in fcfs],
]
pprint(ans_9, width=106)


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
# Equation for the IRR and its real solutions
initial_investment = abs(fcfs[0])
cash_flow = fcfs[1:]
eq, sol = irr_equation(initial_investment, cash_flow)
print(latex(eq))
print(sol)


# %%
# To get the correct answer, I need more precision.
# So I'll also solve the IRR equation by using Newton's method.
def evaluate_polynomial(coefficients, x):
    result = 0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def differentiate_polynomial(coefficients):
    if len(coefficients) < 2:
        return [0]
    else:
        return [coeff * (len(coefficients) - 1 - index) for index, coeff in enumerate(coefficients[:-1])]


def newton_method_polynomial(coefficients, initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    for iteration in range(1, max_iterations + 1):
        fx = evaluate_polynomial(coefficients, x)
        if abs(fx) < tolerance:
            return x, iteration
        dfx = evaluate_polynomial(differentiate_polynomial(coefficients), x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method cannot proceed.")
        x -= fx / dfx
    raise ValueError(f"Newton's method did not converge after {max_iterations} iterations.")


# %%
coefficients = [14600, 83933, 196799, 235776, 142550, 28591, -16128]
initial_guess = 0.2
try:
    root, num_iterations = newton_method_polynomial(coefficients, initial_guess, tolerance=1e-7)
    print(f"Approximate root: {root}")
    print(f"Number of iterations: {num_iterations}")
except ValueError as e:
    print(e)


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
# Compute the net present value
discount_rate = 0.063375
time_periods = 6
ans_11 = npv(initial_investment, cash_flow, discount_rate, time_periods)
ans_11 = int(round(ans_11, 0))
print(format_as_money(ans_11))

# %%
# Payback time
tot = 0
for i, cf in enumerate(cash_flow, start=1):
    tot += cf
    if tot >= initial_investment:
        print(i)
        break


# %%
def mean(vals: list[int]) -> float:
    return sum(vals) / len(vals)


# %%
# Compute the average net income
mean_income = mean(net_income[1:])
print(f"${mean_income:,.2f}")

# %%
# Compute the average of net fixed assets
net_fixed_assets = [350000, 315000, 280000, 245000, 210000, 175000, 140000]
mean_fixed_assets = mean(net_fixed_assets)
print(f"${mean_fixed_assets:,.2f}")

# %%
# Compute the return on invested capital
roic = mean_income / mean_fixed_assets
print(roic)
print(f"{100 * roic}%")
