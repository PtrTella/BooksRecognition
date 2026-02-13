"""Test impatto preprocessing sui keypoints"""
import cv2
import numpy as np

# Test preprocessing con/senza morph
img = cv2.imread('./dataset/models/model_19.png')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l = lab[:, :, 0]

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Versione 1: Solo CLAHE + sharpen (originale)
l1 = clahe.apply(l)
blur = cv2.GaussianBlur(l1, (0, 0), sigmaX=2.0)
l1 = cv2.addWeighted(l1, 1.5, blur, -0.5, 0)

# Versione 2: Con bilateral + morph (nuovo)
l2 = cv2.bilateralFilter(l, d=5, sigmaColor=50, sigmaSpace=50)
l2 = clahe.apply(l2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
l2 = cv2.morphologyEx(l2, cv2.MORPH_CLOSE, kernel)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
l2 = cv2.morphologyEx(l2, cv2.MORPH_OPEN, kernel_small)
blur2 = cv2.GaussianBlur(l2, (0, 0), sigmaX=2.0)
l2 = cv2.addWeighted(l2, 1.5, blur2, -0.5, 0)

# SIFT con parametri aggressivi
sift = cv2.SIFT_create(contrastThreshold=0.01)

kp1, des1 = sift.detectAndCompute(l1, None)
kp2, des2 = sift.detectAndCompute(l2, None)

print(f'SOLO CLAHE+Sharpen: {len(kp1)} keypoints')
print(f'CON Bilateral+Morph: {len(kp2)} keypoints')
print(f'Differenza: {len(kp1) - len(kp2)} ({(len(kp1)-len(kp2))/len(kp1)*100:.1f}% in meno)')

# Test anche ORB
orb = cv2.ORB_create(nfeatures=2000)
kp_orb1, _ = orb.detectAndCompute(l1, None)
kp_orb2, _ = orb.detectAndCompute(l2, None)
print(f'\nORB CLAHE+Sharpen: {len(kp_orb1)} keypoints')
print(f'ORB Bilateral+Morph: {len(kp_orb2)} keypoints')

# Test AKAZE
akaze = cv2.AKAZE_create()
kp_akaze1, _ = akaze.detectAndCompute(l1, None)
kp_akaze2, _ = akaze.detectAndCompute(l2, None)
print(f'\nAKAZE CLAHE+Sharpen: {len(kp_akaze1)} keypoints')
print(f'AKAZE Bilateral+Morph: {len(kp_akaze2)} keypoints')
