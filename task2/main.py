import pandas as pd
import scipy.stats as stats
import math

from scipy.stats import kruskal

def kruskal_wallis_test(group1, group2):
    statistic, p_value = kruskal(group1, group2)
    return statistic, p_value


def anova_test(group1, group2):
    group_data = [group1, group2]

    group_means = [group.mean() for group in group_data]
    grand_mean = sum(group_means) / len(group_means)

    SSB = sum([len(group) * (group_mean - grand_mean) ** 2 for group, group_mean in zip(group_data, group_means)])
    SST = sum([(value - grand_mean) ** 2 for group in group_data for value in group])
    SSW = sum([(value - group_mean) ** 2 for group, group_mean in zip(group_data, group_means) for value in group])

    df_between = len(group_data) - 1
    df_within = sum([len(group) - 1 for group in group_data])

    MSB = SSB / df_between
    MSW = SSW / df_within

    F_statistic = MSB / MSW
    p_value = stats.f.sf(F_statistic, df_between, df_within)

    return F_statistic, p_value
    mean_group1 = group1.mean()
    mean_group2 = group2.mean()
    std_group1 = group1.std()
    std_group2 = group2.std()

    s_pooled = math.sqrt(((len(group1) - 1) * std_group1 ** 2 + (len(group2) - 1) * std_group2 ** 2) / (
            len(group1) + len(group2) - 2))

    t_statistic = (mean_group1 - mean_group2) / (s_pooled * math.sqrt(1 / len(group1) + 1 / len(group2)))

    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), len(group1) + len(group2) - 2))

    return t_statistic, p_value


alpha = 0.05

data = pd.read_csv('../cars93.csv')

usa_prices = data[data['Origin'] == 'USA']['Price']
non_usa_prices = data[data['Origin'] == 'non-USA']['Price']

t_statistic_prices, p_value_prices = anova_test(usa_prices, non_usa_prices)

print("Тест для цен:")
print("Значение t-статистики ANOVA:", t_statistic_prices)
print("Значение p-value:", p_value_prices)

if p_value_prices < alpha:
    print("Отвергаем нулевую гипотезу: различия в ценах статистически значимы для всех трех групп.")
else:
    print("Не отвергаем нулевую гипотезу: различия в ценах не статистически значимы для всех трех групп.")

usa_horsepower = data[data['Origin'] == 'USA']['Horsepower']
non_usa_horsepower = data[data['Origin'] == 'non-USA']['Horsepower']

t_statistic_horsepower, p_value_horsepower = anova_test(usa_horsepower, non_usa_horsepower)

print("\nТест для мощности:")
print("Значение t-статистики ANOVA:", t_statistic_horsepower)
print("Значение p-value:", p_value_horsepower)

if p_value_horsepower < alpha:
    print("Отвергаем нулевую гипотезу: различия в мощности статистически значимы для всех трех групп.")
else:
    print("Не отвергаем нулевую гипотезу: различия в мощности не статистически значимы для всех трех групп.")

print("\nТест для цен:")
print("Значение статистики Краскела-Уоллиса:", t_statistic_prices)
print("Значение p-value:", p_value_prices)

if p_value_prices < alpha:
    print("Отвергаем нулевую гипотезу: различия в ценах статистически значимы для всех трех групп.")
else:
    print("Не отвергаем нулевую гипотезу: различия в ценах не статистически значимы для всех трех групп.")

t_statistic_horsepower, p_value_horsepower = kruskal_wallis_test(usa_horsepower, non_usa_horsepower)

print("\nТест для мощности:")
print("Значение статистики Краскела-Уоллиса:", t_statistic_horsepower)
print("Значение p-value:", p_value_horsepower)

if p_value_horsepower < alpha:
    print("Отвергаем нулевую гипотезу: различия в мощности статистически значимы для всех трех групп.")
else:
    print("Не отвергаем нулевую гипотезу: различия в мощности не статистически значимы для всех трех групп.")
