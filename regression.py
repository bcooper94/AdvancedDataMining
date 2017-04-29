import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hrEmployeeData = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
xAxis = 'JobInvolvement'
yAxis = 'MonthlyRate'
print(hrEmployeeData['WorkLifeBalance'])

promotionVsHourly = hrEmployeeData.plot(x=xAxis, y=yAxis, style='x')
promotionVsHourly.set_xlim(0, hrEmployeeData[xAxis].max() * 1.15)
promotionVsHourly.set_ylim(0, hrEmployeeData[yAxis].max() * 1.15)
# plt.plot(hrEmployeeData['YearsSinceLastPromotion'], hrEmployeeData['HourlyRate'], 'x')
plt.show()
