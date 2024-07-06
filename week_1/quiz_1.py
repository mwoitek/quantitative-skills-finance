# %%
def bond_value(face_value: float, coupon: float, discount_rate: float, maturity: int) -> float:
    pv_1 = face_value / (1 + discount_rate) ** maturity
    pv_2 = coupon * sum((1 + discount_rate) ** (-t) for t in range(1, maturity + 1))
    pv = pv_1 + pv_2
    return pv


# %%
face_value = 1000
coupon = 50
discount_rate = 0.1
maturity = 2
ans_9 = bond_value(face_value, coupon, discount_rate, maturity)
print(round(ans_9, 2))

# %%
discount_rate = 0.03
ans_10 = bond_value(face_value, coupon, discount_rate, maturity)
print(round(ans_10, 2))
