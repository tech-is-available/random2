import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

scores = np.array([72, 88, 64, 74, 67, 79, 85, 75, 89, 77])

# Hypothesized population mean
mu = 70

t_statistic, p_value = stats.ttest_1samp(scores, mu)

n = len(scores)
df = n - 1

alpha = 0.05

print("One Sample T-Test Results:")
print("T-statistic",t_statistic)
print("P-value",p_value)
print("Degrees of Freedom",df)


plt.plot(scores, marker='o')
plt.title('Student Scores')
plt.xlabel('Student Number')
plt.ylabel('Score')
plt.show()


if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("Interpretation: The mean student score is significantly different from 70.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("Interpretation: The mean student score is not significantly different from 70.")

