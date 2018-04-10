
# coding: utf-8

# In[1]:


import cv2
import glob
import numpy as np
from  Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
from scipy.spatial import distance

#[x,y] for each left and right click event will be stored here
rc=list()
rc2=list()

#this function will be called whenever the mouse is right/left clicked

def create_callback(img_index):
    def mouse_callback2(event2, x2, y2, flags2, params2):
    #left-click event value is 1
#     print("EVENT: %s" % event2)
        if event2==1:
            global rc    
            #store the coordinates of the left-click event
            rc.append(["point1",x2,y2,images[img_index][0]])
            cv2.circle(images[img_index][1],(x2,y2), 10, (0,0,255))
            #verify that the mouse click has been registered
            #print (rc)
        #right-click event value is 2
        if event2==2:
            global rc2
            #store the coordinates of the right-click event
            rc2.append(["point2",x2,y2,images[img_index][0]])
            #verify that the mouse click has been registered
            #print (rc2)
    return mouse_callback2

# Asking for user name and image directory
root = Tk()
root = Tk()
root.geometry("600x100")
def retrieve_input():
    global user_name
    global user_occupation
    user_name=textBox.get("1.0","end-1c")
    user_occupation=textBox1.get("1.0","end-1c")
w = Label(root, text="Hello, Please enter your name in the form: FirstName_LastName")
w1 = Label(root, text="Your Occupation")
textBox=Text(root, height=1, width=20)
textBox1=Text(root, height=1, width=20)
w.pack()
textBox.pack()
w1.pack()
textBox1.pack()
buttonCommit=Button(root, height=1, width=20, text="Save",command=lambda:retrieve_input())
buttonCommit.pack()
root.directory = tkFileDialog.askdirectory()
address=root.directory+"/*.jpg"
root.mainloop()
        
#importing all images from the folder
images = [(file, cv2.imread(file,-1)) for file in glob.glob(r"{}".format(address))]
scenes=len(images)

#opening relevent images:
for i, j in zip([4,10], [12,16]):
    
    img1 = images[i][1]
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 640, 480)
    cv2.moveWindow('image1', 20,20)
    
    img2=images[j][1]
    cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image2', 640, 480)
    cv2.moveWindow('image2', 680,20)

    #set mouse callback function for window
    #cv2.setMouseCallback('image1', mouse_callback)
    cv2.setMouseCallback('image1', create_callback(i) )
    cv2.imshow('image1', img1)
    #cv2.setMouseCallback('image2', mouse_callback)
    cv2.setMouseCallback('image2', create_callback(j) )
    cv2.imshow('image2', img2)
    k2=cv2.waitKey(0)
    if k2==27:
        cv2.destroyWindow('image1')
    k2=cv2.waitKey(0)
    if k2==27:
        cv2.destroyWindow('image2')

# Reading camera_v2 text file from Visual SFM output
text_file = open("cameras_v2.txt", "r")
lines = text_file.read().split('\n')

#reading header information
numberofimages=int(lines[16])
text_file.close()

#Reading the image address, names, Rotation matrices(RM), Camera , Focal Length(FL) into seperate lists
imagenames=[None]*numberofimages
CameraPosition=[None]*numberofimages
R_all=[None]*numberofimages
FL=[None]*numberofimages
for i in range(numberofimages):
    imagenames[i]=lines[19+14*i]
    CameraPosition[i]=lines[23+14*i]
    R_all[i]=[lines[26+14*i],lines[27+14*i],lines[28+14*i]]
    FL[i]=lines[20+14*i]

# Storing Rotation matrices in proper float format
RM = []
tempp=[]
for i in range(len(R_all)):
    for k in range(3):
        for j in range(3):
            temporary=float((R_all[i][k].split())[j])
            tempp.append(temporary)
RM = [tempp[x:x+9] for x in range(0, len(tempp),9)]
     
#Fetching rot matrices for point one
index_rot_point1=[]
for i in range(len(rc)):
    index_rot_point1.append([index for index in range(len(imagenames)) if re.sub(r'\W+', '',imagenames[index]) == re.sub(r'\W+', '', rc[i][3])])

#Fetching Camera positions:
Camera=[]
adad=[]
for i in range(len(R_all)):
        for j in range(3):
            ad=(float(CameraPosition[i].split()[j]))
            adad.append(ad)
Camera = [adad[x:x+3] for x in range(0, len(adad),3)]

results=[]
residuals=[]
for pt in [rc,rc2]:
    RL=[] #Rotation matrices
    CX=[] #Camera positions
    fl=[] #focal lengths
    kp=[] #key points
#     residue=[] #residual
    
    # AX=B matrix form
    A=np.empty([2*len(index_rot_point1),3])
    B=np.empty([2*len(index_rot_point1),1])

    for i in range(len(index_rot_point1)):
    # RL=Rotation matrices list for the images selected
        R111=RM[int(index_rot_point1[i][0])]
        RL.append(R111)
    # CX=Camera positions list for the images selected
        X111=Camera[int(index_rot_point1[i][0])]
        CX.append(X111)
    # fl=Focal length list for the images selected
        f111=(float(FL[int(index_rot_point1[i][0])])*0.002)
        fl.append(f111)
    # Key point coordinates list    
        x1=(-(pt[i][1])+1800)*2*0.001
        y1=(-(pt[i][2])+1200)*2*0.001
        kpp=[x1,y1]
        kp.append(kpp)

    # Matrix A
        A11=(RL[i][6]*kp[i][0]+fl[i]*RL[i][0])
        A12=(RL[i][7]*kp[i][0]+fl[i]*RL[i][1])
        A13=(RL[i][8]*kp[i][0]+fl[i]*RL[i][2])
        A21=(RL[i][6]*kp[i][1]+fl[i]*RL[i][3])
        A22=(RL[i][7]*kp[i][1]+fl[i]*RL[i][4])
        A23=(RL[i][8]*kp[i][1]+fl[i]*RL[i][5])

    # MAtrix B
        B11=CX[i][0]*(RL[i][6]*kp[i][0]+fl[i]*RL[i][0])
        B12=CX[i][1]*(RL[i][7]*kp[i][0]+fl[i]*RL[i][1])
        B13=CX[i][2]*(RL[i][8]*kp[i][0]+fl[i]*RL[i][2])
        B21=CX[i][0]*(RL[i][6]*kp[i][1]+fl[i]*RL[i][3])
        B22=CX[i][1]*(RL[i][7]*kp[i][1]+fl[i]*RL[i][4])
        B23=CX[i][2]*(RL[i][8]*kp[i][1]+fl[i]*RL[i][5])

        A[i*2]=[A11,A12,A13]
        A[i*2+1]=[A21,A22,A23]
        B[i*2]=[B11+B12+B13]
        B[i*2+1]=[B21+B22+B23]

# XYZ= 3D coordinates of the two points in the local coordinate system       
    XYZ=np.linalg.lstsq(A, B)[0]
    results.append(XYZ)
    residue_temp=np.matmul(A,XYZ)
    residue=np.subtract(residue_temp,B)
    residuals.append(residue)

dist_3d = distance.euclidean(results[0],results[1])
print("The 3D distance is:")
print(dist_3d)
act_dist=0.985
scl_fct=act_dist/dist_3d
print("The scaling factor using image based approach is:")
print(scl_fct)

from laspy.file import File
from scipy.spatial import distance
import numpy as np

# Open a file in read mode:
inFile=File("CAM_Dsense - las.las")

# Finding the neighbourhood points
dataset=np.vstack([inFile.x, inFile.y, inFile.z]).transpose()

# Finding all points in the range of +-p in x,y,z directions
p=1
index_pos=np.where((dataset[:,0]>=results[0][0]-p)&(dataset[:,0]<=results[0][0]+p)&(dataset[:,1]>=results[0][1]-p)&(dataset[:,1]<=results[0][1]+p)&(dataset[:,2]>=results[0][2]-p)&(dataset[:,2]<=results[0][2]+p))

# Finding the distances of all points in the neighbourhood to the point in question and finding the closest one
dst=[]
points = []
aa=(XYZ[0][0],XYZ[1][0],XYZ[2][0])
minx = 0
miny = 0
minz = 0
min_dist =999999
mini = 0
    
for i in range(len(index_pos[0])):
     if (distance.euclidean(dataset[index_pos[0][i]],aa))<min_dist:
            min_dist = distance.euclidean(dataset[index_pos[0][i]],aa)
            minx = dataset[index_pos[0][i]][0]
            miny = dataset[index_pos[0][i]][1]
            minz = dataset[index_pos[0][i]][2]
            mini = i                 

# Keeping all points in 0.5 unit distance radius
pos=dataset[index_pos[0][mini]]
distances=np.sum((dataset-pos)**2, axis=1)
keep_points=distances<0.5
points_kept=inFile.points[keep_points]
print("We're keeping %i points out of %i total in the new AOI point cloud"%(len(points_kept), len(inFile)))

# Creating a webpage and naming it according to the user_name
trimmedLAS=user_name+'.las'
htmlDOC=user_name+'.html'

outFile=File(trimmedLAS, mode="w", header=inFile.header)
outFile.points=points_kept
outFile.close()

# Creating a text file log of all variables and observations
save_path = 'C:/CrowdData/'
name_of_file = user_name
completeName = save_path + name_of_file+".txt"    
file1 = open(completeName, "w")
toFile = ("\bUser Name:\b \n" + str(user_name) + "\n\bUser Occupation:\b \n" + str(user_occupation) + "\n\bPoint1 Data:\b0 \n" + str(rc) + "\n\bPoint2 Data:\b \n" + str(rc2) + "\n\b3D Point1:\b \n" + str(results[0]) +"\n\b3D Point2:\b \n"+ str(results[1]) + "\n\b3D distance in point cloud:\b \n" + str(dist_3d) + "\n\bActual distance:\b \n" + str(act_dist) + "\n\bScaling Factor:\b \n" + str(scl_fct)) 
file1.write(toFile)
file1.close()

# Running potreeconverster CMD command from jupyter
import os
os.popen("C:\PotreeConverter_1.5_windows_x64\PotreeConverter.exe C:\jupyterNotebook\{0} -o C:/xampp/htdocs/potree --generate-page {1}".format(trimmedLAS,user_name))

# Creating the local potree webpage  
html_address=str("http://localhost/potree/{0}".format(htmlDOC))
print(html_address)

# *** PLEASE CLICK ON THE LINK BELOW ***

