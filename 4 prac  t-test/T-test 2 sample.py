import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

test1 = np.array([85, 68, 67, 84, 98, 60, 94, 80, 94, 98, 95, 80, 85, 87, 75])
test2 = np.array([70, 90, 80, 89, 88, 86, 78, 87, 90, 86, 92, 94, 99, 93, 86])

t_statistic, p_value = stats.ttest_rel(test1, test2)

n = len(test1)
df = n - 1

alpha = 0.05

print("Paired T-Test Results:")
print("T-statistic",t_statistic)
print("P-value",p_value)
print("Degrees of Freedom",df)

plt.plot(test1, marker='o', label='Before Remedial')
plt.plot(test2, marker='o', label='After Remedial')

plt.title('Test Scores Before and After Remedial Lectures')
plt.xlabel('Student Number')
plt.ylabel('Test Score')
plt.legend()
plt.show()

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("Interpretation: There is a significant difference in the mean test scores before and after remedial lectures.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("Interpretation: There is no significant difference in the mean test scores before and after remedial lectures.")
