import time
import warnings
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit, optimize
import lmfit
import scipy

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

current_y = [abs(i) for i in current_y]
print(current_y)
voltagea = voltage_x[0:9]
currenta = current_y[0:9]
p2a = np.poly1d(np.polyfit(voltagea, currenta, 8))

x_werte = voltage_x[0:8]

y = 2 * 10 ** -10 * x_werte ** 2 + 10 ** -10 * x_werte + 4 * 10 ** -11

plt.plot(x_werte, y)

voltageb = voltage_x[8:]
currentb = current_y[8:]
p2b = np.poly1d(np.polyfit(voltageb, currentb, 7))

plt.plot(voltage_x, current_y, color='black', marker='o')

plt.plot(voltagea, p2a(currenta), color='red')
plt.plot(voltageb, p2b(currentb), color='blue')
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

wavelength = []
for child in root.findall('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep'):
    wavelength.append(child.attrib)
    for i in child:
        wavelength.append(list(map(float, i.text.split(','))))

result = []
x = [1, 4, 7, 10, 13, 16, 19]
for i in x:
    result.append(np.array(wavelength[i + 1][0:6065]) - np.array(wavelength[20][0:6065]))

for i in x:
    print(len(wavelength[i]))

print(len(result), 'hier')

print(wavelength[0]['DCBias'])
print(len(wavelength))
plt.plot(wavelength[1], result[0], label=wavelength[0]['DCBias'])
plt.plot(wavelength[4], result[1], label=wavelength[3]['DCBias'])
plt.plot(wavelength[7], result[2], label=wavelength[6]['DCBias'])
plt.plot(wavelength[10], result[3], label=wavelength[9]['DCBias'])
plt.plot(wavelength[13], result[4], label=wavelength[12]['DCBias'])
plt.plot(wavelength[16], result[5], label=wavelength[15]['DCBias'])
plt.plot(wavelength[19], wavelength[20], label=wavelength[18]['DCBias'], color='black')
plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

plt.plot(wavelength[1], wavelength[2], label=wavelength[0]['DCBias'])
plt.plot(wavelength[4], wavelength[5], label=wavelength[3]['DCBias'])
plt.plot(wavelength[7], wavelength[8], label=wavelength[6]['DCBias'])
plt.plot(wavelength[10], wavelength[11], label=wavelength[9]['DCBias'])
plt.plot(wavelength[13], wavelength[14], label=wavelength[12]['DCBias'])
plt.plot(wavelength[16], wavelength[17], label=wavelength[15]['DCBias'])
plt.plot(wavelength[19], wavelength[20], label=wavelength[18]['DCBias'], color='black')

plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

x = np.array(wavelength[19])
y = np.array(wavelength[20])

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
n_max = y.argmax()
plt.plot(x[n_max], y[n_max], 'o')
n_min = y.argmin()
plt.plot(x[n_min], y[n_min], 'o')
plt.show()

r2_2 = r2_score(y, p2(x))
r2_4 = r2_score(y, p(x))
r2_30 = r2_score(y, p30(x))

print('2th degree:', r2_2 * 100)
print('4th degree:', r2_4 * 100)
print('30th degree:', r2_30 * 100)

liste = list(range(0, len(wavelength), 3))
for i in liste:
    plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])

plt.show()

xdata = np.array(wavelength[19])
ydata = np.array(wavelength[20])
liste_p2 = np.poly1d(np.polyfit(xdata, ydata, 2))
plt.plot(x, y, linewidth=0.5)

p2 = liste_p2(ydata)
n_max = p2.argmax()
plt.plot(xdata[n_max], liste_p2[n_max], 'o')
n_min = p2.argmin()
plt.plot(xdata[n_min], liste_p2[n_min], 'o')

plt.show()

# plot iv measurement
# load varible
x_daten = np.array(voltage)
y_daten = np.array(current)
y_daten = [abs(i) for i in current_y]

x_werte_1 = x_daten[0:8]

y_werte_1 = 2 * 10 ** -10 * x_werte ** 2 + 10 ** -10 * x_werte + 4 * 10 ** -11

plt.plot(x_daten, y_daten, 'x', label='Punkte')
plt.plot(x_werte_1, y_werte_1)


# Optimize
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


# p0 = (0.0, 1e-10, -0.25)

x_werte_2 = x_daten[8:12]
y_werte_2 = y_daten[8:12]

params, cv = scipy.optimize.curve_fit(monoExp, x_werte_2, y_werte_2)
m, t, b = params
print(params)


#lmfit
mod = lmfit.models.ExponentialModel()
pars = mod.guess(y_werte_2, x=x_werte_2)
out = mod.fit(y_werte_2, pars, x=x_werte_2)

print(out.fit_report())


plt.plot(x_werte_2, y_werte_2, '.', label="data")
plt.plot(x_werte_2, monoExp(x_werte_2, m, t, b), '--', label="fitted")
plt.plot(x_werte_2, out.best_fit, label = 'best fit')
plt.plot(x_werte_2, out.init_fit, label = 'inti fit')


xFit = np.arange(0, 1, 0.001)

plt.legend()
plt.yscale('log')
plt.show()
