import cv2
import os
import numpy as np

from math import pi, sin, cos
 
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PTAUchar, CPTAUchar
from direct.task import Task
from direct.actor.Actor import Actor

from direct.interval.IntervalGlobal import *
from panda3d.egg import EggData, EggVertexPool, EggVertex, EggGroup, EggLine, loadEggData, EggNurbsCurve
from direct.directutil.Mopath import Mopath
from direct.interval.MopathInterval import *

from direct.task import Task

import numpy as np

from panda3d.core import * 
import random

from pandac.PandaModules import Texture, TextureStage, CardMaker
from direct.gui.OnscreenImage import OnscreenImage

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
def getDes(image, nFeatures=1500):
    orb = cv2.ORB(nFeatures)
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(image,None)
    if kp is None:
        return [],[]
    return orb.compute(image, kp)  #returns pair kp, des

    
def getMatches(des_marker, des_image):
    if des_marker is None or des_image is None:
        return None
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks=50)   # or pass empty dictionary    
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    #print len(des_image), len(des_marker)
    matches1to2 = matcher.knnMatch(des_image,des_marker,k=2)
    matches2to1 = matcher.knnMatch(des_marker,des_image,k=2)

    #ratio test
    matches1to2 = [x for x in matches1to2 if len(x) == 2]
    matches2to1 = [x for x in matches2to1 if len(x) == 2]
    good1to2 = [m for m,n in matches1to2 if m.distance < 0.8*n.distance]
    good2to1 = list([m for m,n in matches2to1 if m.distance < 0.8*n.distance])

    #symmetry test
    good = []
    for m in good1to2:
        for n in good2to1:
            if m.queryIdx == n.trainIdx and n.queryIdx == m.trainIdx:
                good.append(m)
    print 'num matches: ', len(good)
    return good

w = 640
h = 480

win_w = 800
win_h = 600

useCamera = True

class createNurbsCurve():
    def __init__(self):
        self.data = EggData()
        self.vtxPool = EggVertexPool('mopath')
        self.data.addChild(self.vtxPool)
        self.eggGroup = EggGroup('group')
        self.data.addChild(self.eggGroup)
        self.myverts=[]
    
    def addPoint(self,pos): 
        eggVtx = EggVertex()
        eggVtx.setPos(Point3D(pos[0],pos[1],pos[2]))
        self.myverts.append(eggVtx)
        self.vtxPool.addVertex(eggVtx)

    def getNodepath(self):        
        myCurve=EggNurbsCurve()
        myCurve.setup(3, len(self.myverts) +3)
        myCurve.setCurveType(1)
        for i in self.myverts:
            myCurve.addVertex(i)
        self.eggGroup.addChild(myCurve)
        return NodePath(loadEggData(self.data))

class MyApp(ShowBase):
    def __init__(self, markerImage='marker.jpg', calib_file='test.npz'):
        ShowBase.__init__(self)

        base.disableMouse()
        
        self.marker = cv2.imread(markerImage)
        self.marker = cv2.flip(self.marker,0)
        
        self.kp_marker, self.des_marker = getDes(self.marker)
        
        if useCamera:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
        else:
            ret, frame = True, cv2.imread("sample_0.jpg")
            
        if ret:
            self.frame = frame
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        with np.load(calib_file) as calib_File:
            self.K = calib_File['mtx']
            self.D = calib_File['coef']
           
        (h,w) = frame.shape[0:2]
        print w, h

        far = 100
        near = 0.1
        
        fovx, fovy, f, (cx, cy), a = cv2.calibrationMatrixValues(self.K, (w,h), w,h)
        print fovx, fovy, f, cx, cy
        base.camLens.setFilmSize(w, h)
        base.camLens.setFilmOffset(w*0.5 - cx, h*0.5 - cy)
        base.camLens.setFocalLength(f)
        base.camLens.setFov(fovx, fovy)
        base.camLens.setNearFar(near, far)
        
        #base.camLens.setCoordinateSystem(4)
        base.camLens.setCoordinateSystem(4)
        #base.camLens.setViewVector(Vec3(0,0,1), Vec3(0,1,0))
        #self.render.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))

        self.tex = Texture("detect") #self.buff.getTexture()
        self.tex.setCompression(Texture.CMOff) 
        self.tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgb)

        self.b=OnscreenImage(parent=render2d, image=self.tex)
        base.cam.node().getDisplayRegion(0).setSort(20)
 
        self.taskMgr.add(self.updateFrameTask, "UpdateCameraFrameTask")
 
        self.modelroot = NodePath('ARRootNode')
        self.modelroot.reparentTo(self.render)
        
        '''
        self.x = self.loader.loadModel("models/box")
        self.x.reparentTo(self.modelroot)
        self.x.setScale(3, 0.1, 0.1)
        self.x.setPos(0, -0.05, -0.05)
        self.x.setColor(1,0,0,1,1)

        self.y = self.loader.loadModel("models/box")
        self.y.reparentTo(self.modelroot)
        self.y.setScale(0.1, 3, 0.1)
        self.y.setPos(-0.05, 0, -0.05)
        self.y.setColor(0,1,0,1,1)

        self.z = self.loader.loadModel("models/box")
        self.z.reparentTo(self.modelroot)
        self.z.setScale(0.1, 0.1, 3)
        self.z.setPos(-0.05, -0.05, 0)
        self.z.setColor(0,0,1,1,1)
        '''
        
        self.panda = NodePath('PandaRoot')
        self.panda.reparentTo(self.modelroot)
        
        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.003, 0.003, 0.003)
        self.pandaActor.reparentTo(self.panda)
        self.pandaActor.loop("walk")
        self.pandaActor.setH(180)
        
        #self.pandaMotion = MopathInterval("Panda Path", self.panda, "Interval Name")
        
        
        self.pathCurve=createNurbsCurve()
        for i in range(0, 30):
            self.pathCurve.addPoint((random.uniform(1, 7), random.uniform(1, 7), 0))
        '''
        self.pathCurve.addPoint((1, 5, 0))
        self.pathCurve.addPoint((5, 5, 0))
        self.pathCurve.addPoint((5, 1, 0))
        self.pathCurve.addPoint((1, 1, 0))
        '''

        curveNode = self.pathCurve.getNodepath()

        
        self.myMopath = Mopath()
        self.myMopath.loadNodePath(curveNode)
        self.myMopath.fFaceForward = True
        myInterval = MopathInterval(self.myMopath, self.panda, duration=100 ,name = "Name")
        myInterval.loop()
    
        
    # This task runs for two seconds, then prints done
    def updateFrameTask(self, task):
        if useCamera:
            ret, frame = self.cap.read()
        else:
            ret, frame = True, self.frame
         
        if ret:
          frame = cv2.flip(frame,0)
          kp_frame, des_frame = getDes(frame)
          matches = getMatches(self.des_marker, des_frame)
          if not matches or len(matches) < 5:
            return task.cont
          
          pattern_points = [self.kp_marker[pt.trainIdx].pt for pt in matches]
          pattern_points = np.array([(x/50.0,y/50.0,0) for x,y in pattern_points], dtype=np.float32)
          image_points = np.array([kp_frame[pt.queryIdx].pt for pt in matches], dtype=np.float32)
        
          #ret, rvecs, tvecs = cv2.solvePnP(pattern_points, image_points, self.K, None)
          rvecs, tvecs, inliers = cv2.solvePnPRansac(pattern_points, image_points, self.K, None)
          #print "Tadam!",rvecs, tvecs
          #img_pts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)
          #draw_lines(image, img_pts)

          T = tvecs.ravel()
          R = rvecs.ravel()
          
          RotM,_ = cv2.Rodrigues(R)
          RotM = RotM.T
          RotM = Mat3(RotM[0,0],RotM[0,1],RotM[0,2],
                      RotM[1,0],RotM[1,1],RotM[1,2],
#                          RotM[2,0],RotM[2,1],RotM[2,2])
                      -RotM[2,0],-RotM[2,1],-RotM[2,2])
          self.modelroot.setMat(Mat4(RotM, Vec3(T[0],T[1],T[2])))
              
          self.tex.setRamImage(frame)
          self.b.setImage(image=self.tex, parent=render2d)
          
        self.camera.setPos(0,0,0)
        self.camera.setHpr(0,0,0)
        self.camera.setScale(1,1,1)
        
        return task.cont

loadPrcFileData("", "win-size {} {}".format(w,h))
 
app = MyApp()
app.run()
