#
# Approximate AP1 cusp with 6 cos/sin coefficients for ACES2 DRT
#
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

referenceLuminance = 100.0
gamutCuspTableSize = 360

# Hellwig 2022 CAM params
ac_resp = 1.0
surround = (0.9, 0.59, 0.9)

# xy coordintes for custom CAT matrix
rxy = (0.8336, 0.1735)
gxy = (2.3854, -1.4659)
bxy = (0.087, -0.125)
wxy = (0.333, 0.333)

ra = ac_resp * 2
ba = 0.05 + (2.0 - ra)

# Input vars
XYZ_w = (95.05, 100.0, 108.88)  # not used?
XYZ_w_scaler = 100.0
L_A = 100.0
Y_b = 20.0

# Function definitions from Blink
def RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, Y, direction):
# given r g b chromaticities and whitepoint, convert RGB colors to XYZ
# based on CtlColorSpace.cpp from the CTL source code : 77
# param: inverse - calculate XYZ to RGB instead

    r = rxy
    g = gxy
    b = bxy
    w = wxy

    X = w[0] * Y / w[1]
    Z = (1 - w[0] - w[1]) * Y / w[1]

    # Scale factors for matrix rows
    d = r[0] * (b[1] - g[1]) + b[0] * (g[1] - r[1]) + g[0] * (r[1] - b[1])

    Sr =    (X * (b[1] - g[1]) -
            g[0] * (Y * (b[1] - 1.0) +
            b[1]  * (X + Z)) +
            b[0]  * (Y * (g[1] - 1.0) +
            g[1] * (X + Z))) / d

    Sg =    (X * (r[1] - b[1]) +
            r[0] * (Y * (b[1] - 1.0) +
            b[1] * (X + Z)) -
            b[0] * (Y * (r[1] - 1.0) +
            r[1] * (X + Z))) / d

    Sb =    (X * (g[1] - r[1]) -
            r[0] * (Y * (g[1] - 1.0) +
            g[1] * (X + Z)) +
            g[0] * (Y * (r[1] - 1.0) +
            r[1] * (X + Z))) / d

    # Assemble the matrix
    Mdata = np.array([
        Sr * r[0], Sr * r[1], Sr * (1.0 - r[0] - r[1]),
        Sg * g[0], Sg * g[1], Sg * (1.0 - g[0] - g[1]),
        Sb * b[0], Sb * b[1], Sb * (1.0 - b[0] - b[1])
    ])

    newMatrix = np.array([
        [Mdata[0], Mdata[3], Mdata[6]],
        [Mdata[1], Mdata[4], Mdata[7]],
        [Mdata[2], Mdata[5], Mdata[8]],
    ])

    newMatrixInverse = np.linalg.inv(newMatrix)

    # return forward or inverse matrix
    if (direction == 0):
      return newMatrix
    elif (direction == 1):
      return newMatrixInverse

# multiplies a 3D vector with a 3x3 matrix
def vector_dot(m, v):
    return np.dot(m, v)

# convert HSV cylindrical projection values to RGB
def HSV_to_RGB( HSV ):
  C = HSV[2] * HSV[1]
  X = C * (1.0 - abs((HSV[0] * 6.0) % 2.0 - 1.0))
  m = HSV[2] - C

  RGB = np.zeros(3)
  RGB[0] = (C if HSV[0] < 1 / 6 else X if HSV[0] < 2 / 6 else 0 if HSV[0] < 3 /6 else 0 if HSV[0] < 4 / 6 else X if HSV[0] < 5 / 6 else C ) + m
  RGB[1] = (X if HSV[0] < 1 / 6 else C if HSV[0] < 2 / 6 else C if HSV[0] < 3 /6 else X if HSV[0] < 4 / 6 else 0 if HSV[0] < 5 / 6 else 0 ) + m
  RGB[2] = (0 if HSV[0] < 1 / 6 else 0 if HSV[0] < 2 / 6 else X if HSV[0] < 3 /6 else C if HSV[0] < 4 / 6 else C if HSV[0] < 5 / 6 else X ) + m

  return RGB

def spow(base, exponent):
    if(base < 0.0 and exponent != np.floor(exponent)):
      return 0.0
    else:
      return pow(base, exponent)

def float3pow(base, exponent):
      return np.array([pow(base[0], exponent), pow(base[1], exponent), pow(base[2], exponent)])

# "safe" div
def sdiv( a, b ):
    if(b == 0.0):
      return 0.0
    else:
      return a / b

def post_adaptation_non_linear_response_compression_forward(RGB, F_L):
  F_L_RGB = float3pow(F_L * np.abs(RGB) / 100.0, 0.42)
  RGB_c = (400.0 * np.sign(RGB) * F_L_RGB) / (27.13 + F_L_RGB)
  return RGB_c

# basic 3D hypotenuse function, does not deal with under/overflow
def hypot_float3(xyz):
    return np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])

# convert radians to degrees
def degrees( radians ):
    return radians * 180.0 / np.pi

# convert degrees to radians
def radians( degrees ):
    return degrees / 180.0 * np.pi

def XYZ_to_Hellwig2022_JMh(XYZ, XYZ_w, L_A, Y_b, surround):
  XYZ_w = XYZ_w * XYZ_w_scaler
  # Step 0
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  MATRIX_16 = CAT_CAT16
  RGB_w = vector_dot(MATRIX_16, XYZ_w)

  # Always discount illuminant so this calculation is omitted
  # D of 1.0 actually cancels out, so could be removed entirely
  D = 1.0

  # Viewing conditions dependent parameters
  k = 1 / (5 * L_A + 1)
  k4 = pow(k,4)
  F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * pow((1.0 - k4), 2) * pow(5.0 * L_A, 1.0 / 3.0)
  n = Y_b / XYZ_w[1]
  z = 1.48 + np.sqrt(n)

  D_RGB = D * XYZ_w[1] / RGB_w + 1 - D
  RGB_wc = D_RGB * RGB_w
  RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

  # Computing achromatic responses for the whitepoint.
  R_aw = RGB_aw[0]
  G_aw = RGB_aw[1]
  B_aw = RGB_aw[2]

  A_w = ra * R_aw + G_aw + ba * B_aw

  # Step 1
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  RGB = vector_dot(MATRIX_16, XYZ)

  # Step 2
  RGB_c = D_RGB * RGB

  # Step 3
  # Applying forward post-adaptation non-linear response compression.

  RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)

  # Step 4
  # Converting to preliminary cartesian coordinates.
  R_a = RGB_a[0]
  G_a = RGB_a[1]
  B_a = RGB_a[2]
  a = R_a - 12.0 * G_a / 11.0 + B_a / 11.0
  b = (R_a + G_a - 2.0 * B_a) / 9.0

  # Computing the *hue* angle :math:`h`.
  hr = np.arctan2(b, a)
  h = degrees(hr) % 360.0

  # Step 6
  # Computing achromatic responses for the stimulus.
  R_a2 = RGB_a[0]
  G_a2 = RGB_a[1]
  B_a2 = RGB_a[2]

  A = ra * R_a2 + G_a2 + ba * B_a2

  # Step 7
  # Computing the correlate of *Lightness* :math:`J`.
  J = 100.0 * spow(sdiv(A, A_w), surround[1] * z)

  # Step 9
  # Computing the correlate of *colourfulness* :math:`M`.
  M = 43.0 * surround[2] * np.sqrt(a * a + b * b)

  # HK effect block omitted, aas we always have that off

  return np.array([J, M, h])

def post_adaptation_non_linear_response_compression_inverse(RGB, F_L):
  RGB_p =  (np.sign(RGB) * 100.0 / F_L * float3pow((27.13 * np.abs(RGB)) / (400.0 - np.abs(RGB)), 1.0 / 0.42) )
  return RGB_p

def Hellwig2022_JMh_to_XYZ( JMh, XYZ_w, surround, L_A, Y_b):
  J = JMh[0]
  M = JMh[1]
  h = JMh[2]
  XYZ_w = XYZ_w * XYZ_w_scaler
  # Step 0
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  MATRIX_16 = CAT_CAT16
  RGB_w = vector_dot(MATRIX_16, XYZ_w)

  # Always discount illuminant so this calculation is omitted
  # D of 1.0 actually cancels out, so could be removed entirely
  D = 1.0

  # Viewing conditions dependent parameters
  k = 1 / (5 * L_A + 1)
  k4 = pow(k,4)
  F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * pow((1.0 - k4), 2) * pow(5.0 * L_A, 1.0 / 3.0)
  n = Y_b / XYZ_w[1]
  z = 1.48 + np.sqrt(n)

  D_RGB = D * XYZ_w[1] / RGB_w + 1 - D
  RGB_wc = D_RGB * RGB_w
  RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

  # Computing achromatic responses for the whitepoint.
  R_aw = RGB_aw[0]
  G_aw = RGB_aw[1]
  B_aw = RGB_aw[2]

  A_w = ra * R_aw + G_aw + ba * B_aw

  hr = radians(h)

  # HK effect block omitted, aas we always have that off

  # Computing achromatic response :math:`A` for the stimulus.
  A = A_w * spow(J / 100.0, 1.0 / (surround[1] * z))

  # Computing *P_p_1* to *P_p_2*.
  P_p_1 = 43.0 * surround[2]
  P_p_2 = A

  # Step 3
  # Computing opponent colour dimensions :math:`a` and :math:`b`.
  gamma = M / P_p_1
  a = gamma * np.cos(hr)
  b = gamma * np.sin(hr)

  # Step 4
  # Applying post-adaptation non-linear response compression matrix.
  RGB_a = vector_dot(panlrcm, np.array([P_p_2, a, b])) / 1403.0

  # Step 5
  # Applying inverse post-adaptation non-linear response compression.
  RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)

  # Step 6
  RGB = RGB_c / D_RGB

  # Step 7
  MATRIX_INVERSE_16 = np.linalg.inv(CAT_CAT16)
  XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

  return XYZ

def reach_RGB_to_JMh(RGB):
  luminanceRGB = RGB * boundaryRGB * referenceLuminance
  XYZ = vector_dot(RGB_to_XYZ_reach, luminanceRGB)
  JMh = XYZ_to_Hellwig2022_JMh(XYZ, inWhite, L_A, Y_b, surround)
  return JMh

def generate(peakLuminance):
  global AP1corners, AP1CuspTable, primariesLimit, inWhite, boundaryRGB
  global XYZ_to_RGB_reach, RGB_to_XYZ_reach, CAT_CAT16, panlrcm

  XYZ_to_AP1_ACES_matrix = RGBPrimsToXYZMatrix((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), (0.32168, 0.33767), 1.0, 1)
  XYZ_to_RGB_reach = XYZ_to_AP1_ACES_matrix
  RGB_to_XYZ_reach = np.linalg.inv(XYZ_to_RGB_reach)

  CAT_CAT16 = RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, 1.0, 1)

  white = np.array([1.0, 1.0, 1.0])
  inWhite = vector_dot(RGB_to_XYZ_reach, white)
  boundaryRGB = peakLuminance / referenceLuminance

  # Generate the Hellwig2022 post adaptation non-linear compression matrix
  # that is used in the inverse of the model (JMh-to-XYZ).
  panlrcm = np.array([
    [ra, 1.0, ba],
    [1.0, -12.0 / 11.0, 1.0 / 11.0],
    [1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0]
  ])
  panlrcm = np.linalg.inv(panlrcm)

  # Normalize rows so that first column is 460
  for i in range(3):
    n = 460.0 / panlrcm[i][0]
    panlrcm[i] *= n

  # AP1 corner table
  AP1corners = np.zeros((6, 2))
  v = reach_RGB_to_JMh(np.array([1.0, 0.0, 0.0]))
  AP1corners[0][0] = radians(v[2])
  AP1corners[0][1] = v[1]
  v = reach_RGB_to_JMh(np.array([1.0, 1.0, 0.0]))
  AP1corners[1][0] = radians(v[2])
  AP1corners[1][1] = v[1]
  v = reach_RGB_to_JMh(np.array([0.0, 1.0, 0.0]))
  AP1corners[2][0] = radians(v[2])
  AP1corners[2][1] = v[1]
  v = reach_RGB_to_JMh(np.array([0.0, 1.0, 1.0]))
  AP1corners[3][0] = radians(v[2])
  AP1corners[3][1] = v[1]
  v = reach_RGB_to_JMh(np.array([0.0, 0.0, 1.0]))
  AP1corners[4][0] = radians(v[2])
  AP1corners[4][1] = v[1]
  v = reach_RGB_to_JMh(np.array([1.0, 0.0, 1.0]))
  AP1corners[5][0] = radians(v[2])
  AP1corners[5][1] = v[1]

  # AP1 gamut cusp table
  gamutCuspTableUnsorted = np.zeros((gamutCuspTableSize, 3))
  for i in range(gamutCuspTableSize):
    hNorm = float(i) / gamutCuspTableSize
    RGB = HSV_to_RGB([hNorm, 1.0, 1.0])
    gamutCuspTableUnsorted[i] = reach_RGB_to_JMh(RGB)

  minhIndex = 0
  for i in range(1, gamutCuspTableSize):
    if( gamutCuspTableUnsorted[i][2] <  gamutCuspTableUnsorted[minhIndex][2]):
      minhIndex = i

  AP1CuspTable = np.zeros((gamutCuspTableSize, 3))
  for i in range(gamutCuspTableSize):
    AP1CuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize].copy()
    AP1CuspTable[i][2] = radians(AP1CuspTable[i][2])

def f(x, ax, bx, cx, ay, by, cy, off):
    hr = x
    a = np.cos(hr);
    b = np.sin(hr);
    cos_hr2 = a * a - b * b;
    sin_hr2 = 2.0 * a * b;
    cos_hr3 = 4.0 * a * a * a - 3.0 * a;
    sin_hr3 = 3.0 * b - 4.0 * b * b * b;
    M = (ax * a +
         bx * cos_hr2 +
         cx * cos_hr3 +
         ay * b +
         by * sin_hr2 +
         cy * sin_hr3 +
         off)
    return M

def fit_M(x_data, coeffs):
    ax, bx, cx, ay, by, cy, off = coeffs
    M_values = []
    for x in x_data:
        M_values.append(f(x, ax, bx, cx, ay, by, cy, off))
    return np.array(M_values)

def penalized_error_function(coeffs, x_data, reference_data, penalty_factor):
    M_values = fit_M(x_data, coeffs)
    error = np.sum((M_values - reference_data) ** 2)
    penalty = np.sum(np.maximum(0, reference_data - M_values) ** 2)
    return error + penalty_factor * penalty

def scaling_func(x, a, b, c):
    return (a * x) ** b - c

def fit_scaling_curve():
    global a_opt, b_opt, c_opt, ax, bx, cx, ay, by, cy, off

    # Fit for scaling curve
    peakLuminance = np.array([100, 2000, 4000, 8000])
    y_green = []
    for l in peakLuminance:
        generate(l)
        # We use green corner as the target to it against
        y_green.append(AP1corners[2][1])

    y = y_green[0]
    for l in range(len(peakLuminance)):
      y_green[l] /= y - 0.086882   # Small fudge factor to get 100 nits to 1.0 scaling

    result, pcov = curve_fit(scaling_func, peakLuminance, y_green)

    a_opt, b_opt, c_opt = result
    print("Optimized scaling parameters ((ax)^b-c):")
    print(f"a = {a_opt:.5f}, b = {b_opt:.5f}, c = {c_opt:.5f}")

    x_smooth = np.linspace(min(peakLuminance), max(peakLuminance), 100)
    fig, px = plt.subplots(figsize=(10, 6))
    px.scatter(peakLuminance, y_green, label='AP1 green corner')
    px.plot(x_smooth, scaling_func(x_smooth, a_opt, b_opt, c_opt), label='Scaling curve', color='red')
    for x in peakLuminance:
        y = scaling_func(x, a_opt, b_opt, c_opt)
        px.text(x, y, f'({x}, {y:.3f})', fontsize=7, ha='right')
    plt.xlabel('Peak Luminance')
    plt.ylabel('Scaling factor')
    px.legend()
    plt.savefig("aces2_drt_ap1_fit_scaling.png")

def fit_ap1_cusp_plot(peakLuminance):
    global a_opt, b_opt, c_opt, ax, bx, cx, ay, by, cy, off

    generate(peakLuminance)

    x_ap1 = np.split(AP1CuspTable, 3, 1)[2].reshape(-1)
    y_ap1 = np.split(AP1CuspTable, 3, 1)[1].reshape(-1)
    x_corners = np.split(AP1corners, 2, 1)[0].reshape(-1)
    y_corners = np.split(AP1corners, 2, 1)[1].reshape(-1)

    M_values = []
    for x in x_ap1:
        M_values.append(f(x, ax, bx, cx, ay, by, cy, off) *
                        scaling_func(peakLuminance, a_opt, b_opt, c_opt))

    plt.figure(figsize=(8, 8))
    px = plt.subplot(111, projection='polar')
    px.plot(x_ap1, y_ap1, label='AP1 cusp %d nits' % peakLuminance)
    px.plot(x_ap1, M_values, label='AP1 cusp approx 6 coeffs (scaling=%.2f)' %
            scaling_func(peakLuminance, a_opt, b_opt, c_opt))
    px.scatter(x_corners, y_corners, color='red', label='AP1 cusp corners')
    px.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))
    px.set_theta_zero_location('N')
    px.set_theta_direction(-1)
    plt.savefig("aces2_drt_ap1_fit_%d_polar.png" % peakLuminance)

#    fig, px = plt.subplots(figsize=(10, 6))
#    px.plot(x_ap1, y_ap1, label='AP1 cusp %d nits' % peakLuminance)
#    px.plot(x_ap1, M_values, label='AP1 cusp approx 6 coeffs')
#    px.scatter(x_corners, y_corners, color='red', label='AP1 cusp corners')
#    plt.xlabel('hue (rad)')
#    plt.ylabel('M')
#    px.legend()
#    px.grid(True)
#    plt.savefig("aces2_drt_ap1_fit_%d.png" % peakLuminance)

def fit_ap1_cusp(peakLuminance):
    global ax, bx, cx, ay, by, cy, off

    generate(peakLuminance)

    x_ap1 = np.split(AP1CuspTable, 3, 1)[2].reshape(-1)
    y_ap1 = np.split(AP1CuspTable, 3, 1)[1].reshape(-1)
    x_corners = np.split(AP1corners, 2, 1)[0].reshape(-1)
    y_corners = np.split(AP1corners, 2, 1)[1].reshape(-1)

    # Initial guess for the coefficients
    initial_guess = [10.0, 15.0, 8.0, 15.0, -10.0, 8.0, 70.0]

    # Penalty factor, larger values will help getting closer to corners
    penalty_factor = 3.0

    # Fit for AP1 gamut cusp.  Alternative would be to fit to the
    # AP1 corners but this produces better fit.
    result = minimize(penalized_error_function, initial_guess, args=(x_ap1, y_ap1, penalty_factor)).x
    #result, pcov = curve_fit(f, x_ap1, y_ap1, p0=initial_guess)

    # Extract the optimized coefficients
    ax, bx, cx, ay, by, cy, off = result

if __name__ == "__main__":
    fit_scaling_curve()
    fit_ap1_cusp(100)

    print("Optimized cos/sin coefficients:")
    print(f"ax: {ax:.5f}, bx: {bx:.5f}, cx: {cx:.5f}")
    print(f"ay: {ay:.5f}, by: {by:.5f}, cy: {cy:.5f}")
    print(f"off: {off:.5f}")

    # Plot
    peakLuminance = np.array([100, 1000, 2000, 4000, 8000, 10000])
    for n in peakLuminance:
        fit_ap1_cusp_plot(n)

