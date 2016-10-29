#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
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


def draw_model(calibration_filename='test.npz', camera_index=0, model_name='star.stl'):
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
        cap.open(camera_index)
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


def draw_axis(calibration_filename='test.npz', camera_index=0):
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
        cap.open(camera_index)
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


def draw_cube(calibration_filename='test.npz', camera_index=0):
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
        cap.open(camera_index)
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
    try:
        for i, color in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, tuple(img_pts[0].ravel()), tuple(img_pts[i].ravel()), color, 5)
    except OverflowError as er:
        print(er)


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


def get_des(image, n_features=1500):
    orb = cv2.ORB_create(n_features)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(image, None)
    return orb.compute(image, kp)  # returns pair kp, des


def get_matches(des_marker, des_image):
    if des_marker is None or des_image is None:
        return []
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # print len(des_image), len(des_marker)
    matches1to2 = matcher.knnMatch(des_image, des_marker, k=2)
    matches2to1 = matcher.knnMatch(des_marker, des_image, k=2)

    # ratio test
    matches1to2 = [x for x in matches1to2 if len(x) == 2]
    matches2to1 = [x for x in matches2to1 if len(x) == 2]
    good1to2 = [m for m, n in matches1to2 if m.distance < 0.8 * n.distance]
    good2to1 = list([m for m, n in matches2to1 if m.distance < 0.8 * n.distance])

    # symmetry test
    good = []
    for m in good1to2:
        for n in good2to1:
            if m.queryIdx == n.trainIdx and n.queryIdx == m.trainIdx:
                good.append(m)
    'num matches: ', len(good)
    return good


def test_match(marker_name='marker.jpg', camera_index=0):
    cap = cv2.VideoCapture()
    ret, image = cap.read(camera_index)
    marker = cv2.imread(marker_name)

    while ret:
        kp_marker, des_marker = get_des(marker)
        kp_image, des_image = get_des(image)

        good = get_matches(des_marker, des_image)

        # cv2.imshow("image", cv2.drawKeypoints(image,kp_image,color=(0,255,0), flags=0));
        # cv2.imshow("marker", cv2.drawKeypoints(marker,kp_marker,color=(0,255,0), flags=0));
        kp_marker = [kp_marker[pt.trainIdx] for pt in good]
        kp_image = [kp_image[pt.queryIdx] for pt in good]
        cv2.imshow("marker", cv2.drawKeypoints(marker, kp_marker, color=(0, 255, 0), flags=0))
        cv2.imshow("image", cv2.drawKeypoints(image, kp_image, color=(0, 255, 0), flags=0))
        if 27 == cv2.waitKey(1):
            break
        ret, image = cap.read()


def draw_axis_ORB(marker='marker.jpg', calibration_filename='test.npz', camera_index=0):
    marker = cv2.imread(marker)
    kp_marker, des_marker = get_des(marker)

    with np.load(calibration_filename) as X:
        camera_matrix = X['camera_matrix']
        dist_coefs = X['dist_coefs']

        axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        cap = cv2.VideoCapture()
        cap.open(camera_index)
        while True:
            ret, frame = cap.read()
            if ret:
                kp_image, des_image = get_des(frame)

                matches = get_matches(des_marker, des_image)
                if len(matches) > 5:
                    pattern_points = [kp_marker[pt.trainIdx].pt for pt in matches]
                    pattern_points = np.array([(x / 50.0, y / 50.0, 0) for x, y in pattern_points], dtype=np.float32)
                    image_points = np.array([kp_image[pt.queryIdx].pt for pt in matches], dtype=np.float32)

                    _, rvecs, tvecs = cv2.solvePnP(pattern_points, image_points, camera_matrix, dist_coefs)
                    img_pts, jac = cv2.projectPoints(axis, rvecs.ravel(), tvecs.ravel(), camera_matrix, dist_coefs)
                    draw_lines(frame, img_pts)

                cv2.imshow('axes', frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

draw_axis_ORB(camera_index=1)
# first_program()
# create_samples()
# calibrate()
# draw_axis()
# draw_cube()
# draw_model(model_name='Moon.stl')
