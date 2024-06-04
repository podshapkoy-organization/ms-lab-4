import pandas as pd
import scipy.stats as stats
import math

data = pd.read_csv('../cars93.csv')

horsepower = data['Horsepower']
prices = data['Price']

mean_horsepower = horsepower.mean()
mean_price = prices.mean()

covariance = data['Horsepower'].cov(data['Price'])

std_horsepower = horsepower.std()
std_price = prices.std()

pearson_correlation = covariance / (std_horsepower * std_price)

print("Значение коэффициента корреляции Пирсона:", pearson_correlation)

if pearson_correlation > 0:
    print("Есть положительная корреляция между мощностью и ценой: при увеличении мощности цена также увеличивается.")
elif pearson_correlation < 0:
    print("Есть отрицательная корреляция между мощностью и ценой: при увеличении мощности цена уменьшается.")
else:
    print("Нет корреляции между мощностью и ценой: изменение мощности не влияет на цену автомобиля.")

alpha = 0.05
t_statistic_horsepower = pearson_correlation * math.sqrt((len(horsepower) - 2) / (1 - pearson_correlation ** 2))
p_value_horsepower = 2 * (1 - stats.t.cdf(abs(t_statistic_horsepower), len(horsepower) - 2))

print("Значение t-статистики:", t_statistic_horsepower)
print("Значение p-value:", p_value_horsepower)

if p_value_horsepower < alpha:
    print("Отвергаем нулевую гипотезу: мощность влияет на цену автомобиля (коэффициент корреляции не равен 0).")
else:
    print("Не отвергаем нулевую гипотезу: мощность не влияет на цену автомобиля (коэффициент корреляции равен 0).")

spearman_correlation, p_value_spearman = stats.spearmanr(horsepower, prices)

print("Значение коэффициента корреляции Спирмена:", spearman_correlation)

if spearman_correlation > 0:
    print("Есть положительная корреляция между мощностью и ценой: при увеличении мощности цена также увеличивается.")
elif spearman_correlation < 0:
    print("Есть отрицательная корреляция между мощностью и ценой: при увеличении мощности цена уменьшается.")
else:
    print("Нет корреляции между мощностью и ценой: изменение мощности не влияет на цену автомобиля.")

print("Значение p-value:", p_value_spearman)

if p_value_spearman < alpha:
    print("Отвергаем нулевую гипотезу: мощность влияет на цену автомобиля (коэффициент корреляции не равен 0).")
else:
    print("Не отвергаем нулевую гипотезу: мощность не влияет на цену автомобиля (коэффициент корреляции равен 0).")
