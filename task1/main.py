import numpy as np
from scipy.stats import norm, shapiro
import pandas as pd

alpha = 0.05
data = pd.read_csv('../cars93.csv')
prices = data['Price']

sorted_prices = np.sort(prices)

empirical_cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

mu, std = norm.fit(sorted_prices)
normal_cdf = norm.cdf(sorted_prices, mu, std)

max_deviation = np.max(np.abs(empirical_cdf - normal_cdf))

n = len(sorted_prices)
p_value = np.exp(-2 * n * max_deviation ** 2)
print("Значение статистики Колмогорова-Смирнова:", max_deviation)
print("Значение p:", p_value)

if p_value < alpha:
    print("Отвергаем нулевую гипотезу: распределение цены не согласуется с нормальным")
else:
    print("Не отвергаем нулевую гипотезу: распределение цены согласуется с нормальным")

statistic, p_value_shapiro = shapiro(prices)

print("Значение статистики теста Шапиро-Уилка:", statistic)
print("Значение p-value:", p_value_shapiro)

if p_value_shapiro < alpha:
    print("Отвергаем нулевую гипотезу: распределение цены не согласуется с нормальным")
else:
    print("Не отвергаем нулевую гипотезу: распределение цены согласуется с нормальным")
