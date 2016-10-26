import cv2
import os
import numpy as np
from stl import mesh

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboard_size = (7, 9)


def first_program():
    img = cv2.imread('SECR.png')  # Путь к изображению
    cv2.imshow('Image', img)  # Вывод на экран
    cv2.waitKey()  # Ожидаем нажатия клавиши


def create_samples(photos_dir='images/', camera_index=0, show_chess=True):
    """
    Create photos with chessboard for calibration, press 'q' for exit, space for taking photo
    :param photos_dir: dir with photos
    :param camera_index: camera index for OpenCV VideoCapture
    :param show_chess: show image with chessboard detection
    :return: None
    """
    cap = cv2.VideoCapture()
    cap.open(camera_index)
    index = 0
    if not os.path.exists(photos_dir):
        os.mkdir(photos_dir)
    else:
        for file_ in os.listdir(photos_dir):
            os.remove(os.path.join(photos_dir, file_))
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if show_chess:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if ret:
                    cv2.cornerSubPix(gray, corners, (11, 11), (3, 3), criteria)
                    cv2.drawChessboardCorners(gray, chessboard_size, corners, ret)
                    cv2.imshow('chess', gray)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break
            elif c & 0xFF == ord(' '):
                cv2.imwrite(os.path.join(photos_dir, '{}.png'.format(index)), frame)
                index += 1
    cap.release()


def calibrate(photos_dir='images/', result_filename='test.npz'):
    """
    Calibrate and write camera params
    :param photos_dir: dir with photos with chessboard
    :param result_filename: file with calibration params
    :return: None
    """
    pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    pattern_points *= 1
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    images = [os.path.join(photos_dir, file_) for file_ in os.listdir(photos_dir)]
    h, w = 0, 0
    for file_name in images:
        print('processing %s... ' % file_name, end='')
        img = cv2.imread(file_name, 0)
        if img is None:
            print("Failed to load", file_name)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, chessboard_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        else:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
        print('ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    np.savez(result_filename, camera_matrix=camera_matrix, dist_coefs=dist_coefs)


def draw(img, corners, img_pts):
    """
    Draw 3 lines (axises)
    :param img: on this image will be drawing axis
    :param corners: 0,0 point
    :param img_pts: vectors of axises
    :return: img with axises
    """
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)


def draw_model(calibration_filename='test.npz', model_name='star.stl'):
    """
    Find and draw 3D stl model
    :param calibration_filename: file with camera params
    :param model_name: stl model
    :return: None
    """
    with np.load(calibration_filename) as X:
        camera_matrix = X['camera_matrix']
        dist_coefs = X['dist_coefs']
        pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

        cap = cv2.VideoCapture()
        cap.open(0)
        your_mesh = mesh.Mesh.from_file(model_name)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if found:
                    cv2.cornerSubPix(gray, corners, (11, 11), (3, 3), criteria)
                    ret, rvecs, tvecs = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coefs)
                    for vector in your_mesh.vectors:
                        img_pts, jac = cv2.projectPoints(vector, rvecs, tvecs, camera_matrix, dist_coefs)
                        draw_triangle(frame, img_pts)

                    cv2.imshow('chess', frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


def draw_axis(calibration_filename='test.npz'):
    """
    Find and draw 3D axis on chessboard
    :param calibration_filename: file with camera params
    :return: None
    """
    with np.load(calibration_filename) as X:
        camera_matrix = X['camera_matrix']
        dist_coefs = X['dist_coefs']
        pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
        axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        cap = cv2.VideoCapture()
        cap.open(0)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if found:
                    cv2.cornerSubPix(gray, corners, (11, 11), (3, 3), criteria)
                    ret, rvecs, tvecs = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coefs)
                    img_pts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)
                    draw_lines(frame, img_pts)

                    cv2.imshow('chess', frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


def draw_cube(calibration_filename='test.npz'):
    """
    Find and draw 3D axis on chessboard
    :param calibration_filename: file with camera params
    :return: None
    """
    with np.load(calibration_filename) as X:
        camera_matrix = X['camera_matrix']
        dist_coefs = X['dist_coefs']
        pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

        cap = cv2.VideoCapture()
        cap.open(0)
        cube_points = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if found:
                    cv2.cornerSubPix(gray, corners, (11, 11), (3, 3), criteria)
                    ret, rvecs, tvecs = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coefs)
                    # cv2.drawChessboardCorners(gray, chessboard_size, corners, ret)
                    img_pts, jac = cv2.projectPoints(cube_points, rvecs, tvecs, camera_matrix, dist_coefs)
                    draw_cube_model(frame, img_pts)

                    cv2.imshow('chess', frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


def draw_lines(img,  img_pts):
    for i, color in zip(range(1,4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, tuple(img_pts[0].ravel()), tuple(img_pts[i].ravel()), color, 5)


def draw_triangle(img, points):
    points = np.int32(points).reshape(-1, 2)
    cv2.drawContours(img, [points], -1, (0, 255, 0), -3)
    cv2.line(img, tuple(points[0]), tuple(points[1]), 255, 1)
    cv2.line(img, tuple(points[1]), tuple(points[2]), 255, 1)
    cv2.line(img, tuple(points[2]), tuple(points[0]), 255, 1)


def draw_cube_model(img, img_pts):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), -3)
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), 255, 3)
        cv2.drawContours(img, [img_pts[4:]], -1, (0, 0, 255), 3)


first_program()
# create_samples()
# calibrate()
# draw_axis()
# draw_cube()
# draw_model(model_name='Moon.stl')

