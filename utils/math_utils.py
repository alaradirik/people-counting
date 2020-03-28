

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def is_intersection(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def compare_with_prev_position(prev_detection, detection, line, entry, exit):
	(xmin_prev, ymin_prev) = (int(prev_detection[0]), int(prev_detection[1]))
	(xmax_prev, ymax_prev) = (int(prev_detection[2]), int(prev_detection[3]))

	p0 = (
		int(detection[0] + (detection[2]-detection[0])/2), 
		int(detection[1] + (detection[3]-detection[1])/2)
	)
	p1 = (int(xmin_prev + (xmax_prev-xmin_prev)/2), int(ymin_prev + (ymax_prev-ymin_prev)/2))

	d = ((p0[0] - line[0][0])*(line[1][1] - line[0][1]))- ((p0[1] - line[0][1])*(line[1][0] - line[0][0]))

	if is_intersection(p0, p1, line[0], line[1]):
		if d < 0:
			entry += 1
		if d > 0:
			exit += 1

	return entry, exit