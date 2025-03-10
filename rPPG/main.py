import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def estimate_blood_pressure(heart_rate, filtered_signal):
    avg_amplitude = np.mean(filtered_signal)
    systolic = 110 + (heart_rate - 60) * 0.5 + avg_amplitude * 0.1
    diastolic = 70 + (heart_rate - 60) * 0.3 + avg_amplitude * 0.05
    return systolic, diastolic

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

green_signal = []
frame_times = []
fs = 30.0
print("Press 'q' to stop recording...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = frame_rgb[y:y+h//5, x + w//4:x + 3*w//4]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x + w//4, y), (x + 3*w//4, y+h//5), (0, 255, 0), 2)
    else:
        h_frame, w_frame, _ = frame_rgb.shape
        roi = frame_rgb[h_frame//3:2*h_frame//3, w_frame//3:2*w_frame//3]
        cv2.rectangle(frame, (w_frame//3, h_frame//3), (2*w_frame//3, 2*h_frame//3), (0, 255, 0), 2)
    green_avg = np.mean(roi[:, :, 1])
    green_signal.append(green_avg)
    frame_times.append(len(green_signal) / fs)
    cv2.imshow('Video with ROI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

green_signal = np.array(green_signal)
frame_times = np.array(frame_times)
filtered_signal = bandpass_filter(green_signal, lowcut=0.7, highcut=4.0, fs=fs, order=5)
peaks, _ = find_peaks(filtered_signal, distance=fs/2.5)
heart_rate = len(peaks) / (frame_times[-1] / 60)
systolic, diastolic = estimate_blood_pressure(heart_rate, filtered_signal)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(frame_times, green_signal, label='Raw Green Signal')
plt.xlabel('Time (s)')
plt.ylabel('Average Green Value')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(frame_times, filtered_signal, label='Filtered Signal', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Filtered Signal')
plt.legend()
plt.tight_layout()
plt.show()

print("Estimated Heart Rate: {:.2f} BPM".format(heart_rate))
print("Estimated Systolic BP: {:.2f} mmHg".format(systolic))
print("Estimated Diastolic BP: {:.2f} mmHg".format(diastolic))
