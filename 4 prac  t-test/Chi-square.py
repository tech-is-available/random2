import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

aptitude = np.array([85,65,50,68,87,74,65,96,68,94,73,84,85,87,91,117])
jobprof = np.array([70,90,80,89,88,86,78,67,86,90,92,94,99,93,85,2])

chi_statistic, p_value = stats.chisquare(aptitude, jobprof)

n = len(aptitude)
df = n - 1

alpha = 0.05

print("Chi-Square Test Results:")
print("Chi-square statistic", chi_statistic)
print("P-value", p_value)
print("Degrees of Freedom", df)

plt.plot(aptitude, marker='o', label='Aptitude')
plt.plot(jobprof, marker='o', label='Job Proficiency')

plt.title('Aptitude vs Job Proficiency')
plt.xlabel('Employee Number')
plt.ylabel('Score')
plt.legend()
plt.show()

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("Interpretation: There is a significant relationship between aptitude and job proficiency.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("Interpretation: There is no significant relationship between aptitude and job proficiency.")

