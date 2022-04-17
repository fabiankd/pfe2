import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import r2_score
import time

tree = ET.parse('/Users/fabiankading/PycharmProjects/pr2/HY202103_D08_02_LION1_DCM_LMZC.xml')
root = tree.getroot()

values = []
for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
    values.append(child.text)

voltage_string = values[0].split(',')
voltage = []
for i in voltage_string:
    voltage.append(float(i))
print(voltage)

current_string = values[1].split(',')
current = []
for i in current_string:
    current.append(float(i))
print(current)

voltage_x = np.array(voltage)
current_y = np.array(current)

current_y = [i**2 for i in current_y]
current_y = [i*10e9 for i in current_y]
#current_y = [abs(i * 10e9) for i in current_y]

p2 = np.poly1d(np.polyfit(voltage_x, current_y, 2))
print(p2)


plt.plot(voltage_x, current_y, color='black', marker='o')
plt.plot(voltage_x, p2(current_y), color='red')
plt.yscale('log')
plt.show()

current = [abs(i) for i in current]

plt.plot(voltage, current, color='black', marker='o', markeredgecolor='black', markerfacecolor='red')
plt.yscale('log')
plt.title('IV-Analysis')
plt.ylabel('Current in A')
plt.xlabel('Voltage in V')
plt.grid('true')

plt.tight_layout()
plt.show()

wavelenght = []
for child in root.findall('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep'):
    wavelenght.append(child.attrib)
    for i in child:
        wavelenght.append(list(map(float, i.text.split(','))))

result = []
x = [1, 4, 7, 10, 13, 16, 19]
for i in x:
    result.append(np.array(wavelenght[i + 1]) - np.array(wavelenght[20]))

for i in x:
    print(len(wavelenght[i]))

print(len(result), 'hier')

print(wavelenght[0]['DCBias'])
print(len(wavelenght))
plt.plot(wavelenght[1], result[0], label=wavelenght[0]['DCBias'])
plt.plot(wavelenght[4], result[1], label=wavelenght[3]['DCBias'])
plt.plot(wavelenght[7], result[2], label=wavelenght[6]['DCBias'])
plt.plot(wavelenght[10], result[3], label=wavelenght[9]['DCBias'])
plt.plot(wavelenght[13], result[4], label=wavelenght[12]['DCBias'])
plt.plot(wavelenght[16], result[5], label=wavelenght[15]['DCBias'])
plt.plot(wavelenght[19], wavelenght[20], label=wavelenght[18]['DCBias'], color='black')
plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

plt.plot(wavelenght[1], wavelenght[2], label=wavelenght[0]['DCBias'])
plt.plot(wavelenght[4], wavelenght[5], label=wavelenght[3]['DCBias'])
plt.plot(wavelenght[7], wavelenght[8], label=wavelenght[6]['DCBias'])
plt.plot(wavelenght[10], wavelenght[11], label=wavelenght[9]['DCBias'])
plt.plot(wavelenght[13], wavelenght[14], label=wavelenght[12]['DCBias'])
plt.plot(wavelenght[16], wavelenght[17], label=wavelenght[15]['DCBias'])
plt.plot(wavelenght[19], wavelenght[20], label=wavelenght[18]['DCBias'], color='black')

plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

x = np.array(wavelenght[19])
y = np.array(wavelenght[20])

z = np.polyfit(x, y, 4)
p = np.poly1d(z)

start = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))
end = time.time()
print('30nd degree:', end - start, 's')

start = time.time()
p2 = np.poly1d(np.polyfit(x, y, 2))
end = time.time()
print('2nd degree:', end - start, 's')

p2 = np.poly1d(np.polyfit(x, y, 2))

plt.plot(x, y, linewidth=0.5)
plt.plot(x, p(x), 'r--')
plt.plot(x, p2(x), 'g')
plt.plot(x, p30(x), 'b')
plt.show()

r2_2 = r2_score(y, p2(x))
r2_4 = r2_score(y, p(x))
r2_30 = r2_score(y, p30(x))

print('2th degree:', r2_2 * 100)
print('4th degree:', r2_4 * 100)
print('30th degree:', r2_30 * 100)

x = list(range(0, len(wavelenght), 3))
for i in x:
    plt.plot(wavelenght[i + 1], wavelenght[i + 2], label=wavelenght[i]['DCBias'])

plt.show()
