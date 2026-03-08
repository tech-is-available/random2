from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class_A = [85, 90, 88, 82, 87]
class_B = [76, 78, 80, 81, 75]
class_C = [92, 88, 94, 89, 90]

f_statistic, p_value = stats.f_oneway(class_A, class_B, class_C)

print("One-way ANOVA Results:")
print("F-statistic:", f_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis.")
    print("Interpretation: There is a significant difference in mean exam scores among the classes.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("Interpretation: There is no significant difference in mean exam scores among the classes.")

data = class_A + class_B + class_C

labels = (['A'] * len(class_A) +
               ['B'] * len(class_B) +
               ['C'] * len(class_C))

tukey = pairwise_tukeyhsd(data, labels)

print("\nTukey Test Results:")
print(tukey)
