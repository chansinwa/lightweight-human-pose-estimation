import numpy as np
import cv2
import math


def calculate_clockwise_angle_from_x_axis(center, pt):
	"""
	angle_radian: float = 0
	angle_deg: float = 0
		
	if (pt[0] >= center[0] and pt[1] > center[1]):
		angle_radian = math.atan((pt[1] - center[1]) / (pt[0] - center[0]))
		angle_deg = angle_radian * 180 / math.pi
	elif (pt[0] < center[0] and pt[1] >= center[1]):
		angle_radian = math.atan((center[0] - pt[0]) / (pt[1] - center[1]))
		angle_deg = 90 + angle_radian * 180 / math.pi
	elif (pt[0] <= center[0] and pt[1] < center[1]):
		angle_radian = math.atan((center[1] - pt[1]) / (center[0] - pt[0]))
		angle_deg = 180 + angle_radian * 180 / math.pi
	elif (pt[0] > center[0] and pt[1] <= center[1]):
		angle_radian = math.atan((pt[0] - center[0]) / (center[1] - pt[1]))
		angle_deg = 270 + angle_radian * 180 / math.pi

	print("Radian: ", angle_radian, "Degree: ", angle_deg)
	"""
	
	dx = pt[0] - center[0]
	dy = pt[1] - center[1]

	angle_radian = math.atan2(dy, dx)
	angle_deg = math.degrees(angle_radian) % 360
 
	"""
	angle_deg = -math.degrees(math.atan2(a[1] - center[1], a[0] - center[0]))

	if angle_deg < 0:
		angle_deg = 360 + angle_deg
	"""
	return angle_deg

"""
Center:  (626, 277) Point A:  (711, 307) Point B:  (611, 199) X-axis angle:  340.5599651718238 End angle:  120.3255618828349
Center:  (893, 270) Point A:  (806, 305) Point B:  (915, 192) X-axis angle:  74.24882633654697 End angle:  127.66603999064473
Center:  (628, 274) Point A:  (709, 306) Point B:  (615, 195) X-axis angle:  338.44292071496227 End angle:  120.90175118713742
Center:  (891, 267) Point A:  (806, 305) Point B:  (911, 188) X-axis angle:  75.79323388223963 End angle:  128.29421842639263
Center:  (633, 257) Point A:  (709, 305) Point B:  (637, 184) X-axis angle:  327.7243556854224 End angle:  119.13928594624502
Center:  (881, 256) Point A:  (806, 305) Point B:  (882, 177) X-axis angle:  89.27477570094075 End angle:  123.88314818373149
Center:  (635, 254) Point A:  (708, 305) Point B:  (641, 182) X-axis angle:  325.06068979532296 End angle:  120.17566851395088
Center:  (878, 253) Point A:  (806, 305) Point B:  (878, 175) X-axis angle:  90.0 End angle:  125.8376529542783
Center:  (635, 253) Point A:  (708, 305) Point B:  (642, 182) X-axis angle:  324.5366357584594 End angle:  119.83268148390533
Center:  (877, 253) Point A:  (806, 305) Point B:  (877, 175) X-axis angle:  90.0 End angle:  126.21883726561452
Center:  (639, 248) Point A:  (708, 305) Point B:  (663, 177) X-axis angle:  320.4403320310055 End angle:  110.8829662368044
Center:  (873, 249) Point A:  (802, 305) Point B:  (855, 170) X-axis angle:  102.83560948640144 End angle:  115.42837929947919
Center:  (641, 247) Point A:  (708, 305) Point B:  (667, 175) X-axis angle:  319.1181897039888 End angle:  111.02659592669016
Center:  (872, 247) Point A:  (800, 305) Point B:  (852, 169) X-axis angle:  104.3813945910906 End angle:  114.47197974926273
Center:  (646, 242) Point A:  (708, 305) Point B:  (690, 178) X-axis angle:  314.5416435419996 End angle:  100.94983347033201
Center:  (871, 247) Point A:  (800, 305) Point B:  (835, 172) X-axis angle:  115.64100582430528 End angle:  103.60442884459253
Center:  (648, 241) Point A:  (708, 305) Point B:  (693, 179) X-axis angle:  313.1523897340054 End angle:  100.87522321670968
Center:  (871, 247) Point A:  (799, 305) Point B:  (832, 173) X-axis angle:  117.7904424794521 End angle:  101.06293186090123
1Center:  (652, 243) Point A:  (708, 305) Point B:  (705, 183) X-axis angle:  312.0891621738323 End angle:  96.45560428176259
"""

if __name__ == "__main__":
	img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
	# cv2.line(img, (0, 400), (800, 400), (0, 0, 0), 1)
	# cv2.line(img, (400, 0), (400, 800), (0, 0, 0), 1)

	center = (626, 277)
	point_a = (711, 307)
	point_b = (611, 199)
 
	cv2.line(img, center, point_a, (255, 0, 0), 1) # BGR
	cv2.line(img, center, point_b, (0, 255, 0), 1) # BGR

	xaxis_angle: float = 0
	end_angle: float = 0

	angle_a = calculate_clockwise_angle_from_x_axis(center, point_a)
	print("Angle A: ", angle_a)
	angle_b = calculate_clockwise_angle_from_x_axis(center, point_b)
	print("Angle B: ", angle_b)

	if angle_b < angle_a:
		temp = angle_a
		angle_a = angle_b
		angle_b = temp

	if angle_b - angle_a < 180:
		xaxis_angle = angle_a
		end_angle = angle_b - angle_a
	else:
		xaxis_angle = angle_b
		end_angle = 360 - (angle_b - angle_a)

	print("Angle A: ", angle_a, "Angle B: ", angle_b, "X-axis angle: ", xaxis_angle, "End angle: ", end_angle)
	cv2.ellipse(img, center, (30, 30), xaxis_angle, 0, end_angle, (0, 0, 0), 1)

	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
