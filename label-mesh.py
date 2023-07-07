"""
(c) 2023 Eleftherios Ioannou
"""

import nibabel as nib
import numpy as np
import math
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

""" 
This script creates a nodes array nx3 (n: number of nodes) of nodal
coordinates or element centroids.

In the case of nodal points, it appends columns for 5 variables used in the
PROTEAS brain model. The variables are:
1. Healthy cells
2. Tumour cells
3. Necrotic cells
4. Vascular cells
5. Oedema

The variable values are obtained from corresponding images by quering the pixel
value at each nodal coordinate. These are stored into an array (size n) and
appended to the main solution array (nx8 once completed) which is then outputed
(only variable values are output, not coordinates).

> Variable values are set as follows
1. Tumour and necrotic regions are populated first according to segmentation
2. Host cells are all the remaining nodes
3. Vascular cells are in all regions except in the necrotic region
4. Oedema region is set according to segmentation and overlap with other cell regions

> MRI values is obtained from the T1 weighted image in the same way

> Radiation field is obtained from separate image file in the same way

In the case of elements, the centroid of each element is computed from the nodal
coordinates and the appropriate brain region (white/grey matter) is assigned. A
new mesh is output in the end where the elements are labeled according to
whether they belong to the region of white matter (30) or grey matter (40).

"""

### Specify case details
basepath='RP02/'
nifti_brain = basepath+'brain_mask.nii'
nifti_cancer = basepath+'tumour_mask.nii'
nifti_brain_MRI = basepath+'brain_t1.nii'
nifti_brain_RT = basepath+'brain_RD.nii'

msh_file = basepath+'seg_brain.msh'

### Case name (used in output files)
name='RP02-test'
### Specify output
out_dir='' 
warnings=0

### Obtain nodes coordinates from msh file
f_msh = open(msh_file,'r')
lines_msh = f_msh.readlines()
f_msh.close()

nodes_size = int(lines_msh[4])
nodes=np.zeros((nodes_size,3))
for line in lines_msh[5:nodes_size+5]:
    coord = np.fromstring(line, dtype=float, sep=' ')
    nodes[int(coord[0])-1,0]=coord[1]
    nodes[int(coord[0])-1,1]=coord[2]
    nodes[int(coord[0])-1,2]=coord[3]

elements_size = int(lines_msh[nodes_size+7])
elements=np.zeros((elements_size,5),dtype=int)
for line in lines_msh[nodes_size+8:elements_size+nodes_size+8]:
    elem_desc = np.fromstring(line, dtype=int, sep=' ')
    elements[int(elem_desc[0])-1,0]=elem_desc[1] # element type
    elements[int(elem_desc[0])-1,1]=elem_desc[5] # element node
    elements[int(elem_desc[0])-1,2]=elem_desc[6]
    elements[int(elem_desc[0])-1,3]=elem_desc[7]
    if ( elem_desc.size > 8 ): elements[int(elem_desc[0])-1,4]=elem_desc[8]
    
if nodes_size == nodes[:,0].size and elements_size == elements[:,0].size:
    print("Imported ", nodes[:,0].size, " nodes.")
    print("Imported ", elements[:,0].size, " elements.")
else:
    print("ERROR: Number of imported nodes and/or elements does not match nodes specified in msh file")
    exit()

#### Tumour, necrotic, oedema ###

# Load images file
img_brain = nib.load(nifti_brain)
img_cancer = nib.load(nifti_cancer)

# Convert the voxel orientation to RAS
img_brain = nib.as_closest_canonical(img_brain)
print('[Brain image] Voxel orientation is '+str(nib.aff2axcodes(img_brain.affine)))
print('[Brain image] Voxel size', img_brain.header.get_zooms())
print('[Brain image] Dimensions', img_brain.shape)

img_cancer = nib.as_closest_canonical(img_cancer)
#print('[Cancer image] Voxel orientation is '+str(nib.aff2axcodes(img_cancer.affine)))
#print('[Cancer image] Voxel size', img_cancer.header.get_zooms())
#print('[Cancer image] Dimensions', img_cancer.shape)

if img_brain.shape != img_cancer.shape or img_brain.header.get_zooms() != img_cancer.header.get_zooms():
    print("ERROR: Brain and cancer images do not match")
    exit()    
    
### Obtain inverse affine matrix
brain_invaff_mat=np.linalg.inv(img_brain.affine)
cancer_invaff_mat=np.linalg.inv(img_cancer.affine)

# Get data from image to a numpy array
brain_na = img_brain.get_fdata()
cancer_na = img_cancer.get_fdata()
# Plot a slice (for testing purposes)
#slice=brain_na[:,:,66]
#plt.imshow(slice.T, cmap='gray', origin='lower')
#plt.show()

### Populate variables value for each node
hos,tum,nec,oed=[np.zeros((nodes_size,1)) for _ in range(4)]
vsc=np.ones((nodes_size,1))

for idx, node in enumerate(nodes):
    v_pos = cancer_invaff_mat.dot(np.append(node[:3],1))
    label = cancer_na[tuple(v_pos[:3].astype(int))]
    if label == 1:
        nec[idx]=1
        vsc[idx]=0
    elif label == 4:
        tum[idx]=1
    else:
        hos[idx]=1
    if label == 2:
        # print(idx,node,v_pos[:3].astype(int),label)
        oed[idx]=1
        
nodes=np.concatenate((nodes,hos,tum,nec,vsc,oed),axis=1)

### MRI ###

mri=np.zeros((nodes_size,1), dtype=np.double)
if nifti_brain_MRI is None:
    print('WARNING: MRI image not provided. MRI values set to linear dependance on position (x+y+z).')
    warnings+=1
    for idx, node in enumerate(nodes):
        mri[idx] = node[0]+node[1]+node[2]
else:
    img_MRI = nib.load(nifti_brain_MRI)
    # Convert the voxel orientation to RAS
    img_MRI = nib.as_closest_canonical(img_MRI)

    print('[MRI image] Voxel orientation is '+str(nib.aff2axcodes(img_MRI.affine)))
    print('[MRI image] Voxel size', img_MRI.header.get_zooms())
    mri_aff_mat = img_MRI.affine
    mri_invaff_mat=np.linalg.inv(mri_aff_mat)

    # Get data from image to a numpy array
    mri_na = img_MRI.get_fdata()
    print('[MRI image] Dimensions', img_MRI.shape)

    for idx, node in enumerate(nodes):
        v_pos = mri_invaff_mat.dot(np.append(node[:3],1))
        mri[idx] = mri_na[tuple(v_pos[:3].astype(int))]
        if mri[idx] < -1000:
            print('WARNING: Hounsfield value for node '+str(idx)+' is '+str(rd[idx])+'. Set to -999.999.')
            mri[idx]=-999.999
            warnings+=1
            
print('[MRI image] Maximum value:', np.max(mri))
print('[MRI image] Minimum value:', np.min(mri))
nodes=np.append(nodes,mri,axis=1)

### RADIATION ###

ARTIFICIAL_RA=False
rtd=np.zeros((nodes_size,1), dtype=np.double)
if nifti_brain_RT is None:
    print('WARNING: Radiation dosage image not provided. Artificial radiation field is applied.')
    print("INFO: The artificial radiation field is a 3D Gaussian distribution set in the middle of the domain")
    ARTIFICIAL_RA=True
    warnings+=1
else:
    img_RTD = nib.load(nifti_brain_RT)
    # Convert the voxel orientation to RAS
    img_RTD = nib.as_closest_canonical(img_RTD)

    print('[RTD image] Voxel orientation is '+str(nib.aff2axcodes(img_RTD.affine)))
    print('[RTD image] Voxel size', img_RTD.header.get_zooms())
    rtd_aff_mat = img_RTD.affine
    # IMPORTANT: When converting radiation dose dicom to nifty, slice thickness
    # is sometimes not transferred. We set it to 3 mm which is used in BOCOC
    # images but this is not applicaple everywhere
    #if rd_aff_mat[2,2] == 1:
    #   rd_aff_mat[2,2] = 3

    rtd_invaff_mat=np.linalg.inv(rtd_aff_mat)

    # Get data from image to a numpy array
    rtd_na = img_RTD.get_fdata()
    print('[RTD image] Dimensions', img_RTD.shape)
    
    for idx, node in enumerate(nodes):
        v_pos = rtd_invaff_mat.dot(np.append(node[:3],1))
        rtd[idx] = rtd_na[tuple(v_pos[:3].astype(int))]
        # When converting RT DICOM image to nifty using dcm2niix, it was found
        # that it required scalling of the pixel values in order to match Slicer
        # data (scalling was different for some images). 
        if rtd[idx] < 0:
            print('WARNING: Radiation value for node '+str(idx)+' is '+str(rd[idx])+' (negative). Set to 0.')
            rtd[idx]=0
            warnings+=1

# Code for creating mock radiation field if needed
if ARTIFICIAL_RA:

    max_rtd_dose=20
    #### Set position of mean depending on nodes coordinates range
    x_min,x_max=np.min(nodes[:,0]),np.max(nodes[:,0])
    y_min,y_max=np.min(nodes[:,1]),np.max(nodes[:,1])
    z_min,z_max=np.min(nodes[:,2]),np.max(nodes[:,2])
    
    x_var,y_var,z_var=(x_max-x_min),(y_max-y_min),(z_max-z_min)
    x_mean,y_mean,z_mean=x_var/2 + x_min, y_var/4 + y_min, z_var/3 + z_min
    #print(x_max,y_max,z_max)
    
    x_var_coeff,y_var_coeff,z_var_coeff=1,0.5,0.2
    
    for index, node in enumerate(nodes):
        rtd[index]=math.exp(-((node[0]-x_mean)/(x_var*x_var_coeff))**2) * math.exp(-((node[1]-y_mean)/(y_var*y_var_coeff))**2) * math.exp(-((node[2]-z_mean)/(z_var*z_var_coeff))**2)
        rtd[index]*=max_rtd_dose
        
print('[RTD image] Maximum value:', np.max(rtd))
print('[RTD image] Minimum value:', np.min(rtd))
nodes=np.append(nodes,rtd,axis=1)

### Output nodal files

nodal_field_file=out_dir+name+'-nodal_field.dat'
f_out = open(nodal_field_file, 'w')
f_out.write('# hos tum nec vsc oed\n')
for node in nodes:
    f_out.write(' '.join(map(str,node[3:8]))+'\n')
    # output node coordinates as well
    #f_out.write(' '.join(map(str,node))+'\n')
f_out.close()

nodal_field_file_aux=out_dir+name+'-nodal_field_aux.dat'
f_out = open(nodal_field_file_aux, 'w')
f_out.write('# mri rtd\n')
for node in nodes:
    f_out.write(' '.join(map(str,node[8:]))+'\n')
f_out.close()

print("The nodal field has been successfully output at", nodal_field_file, "and", nodal_field_file_aux)

### Label elements

ele_labels=np.zeros((elements_size,1),dtype=int)

relabeled=0
for idx, element in enumerate(elements):    
    if element[0] == 4:
        #print("Element", idx+1, "is", element)
        centroid = np.zeros(3,dtype=float)
        for i in range(1,5):
            # print("Node ", element[i], "has coordinates", nodes[element[i]-1,:3]) # Important: In .msh files, node enumeration starts from 1 but here it starts from 0
            centroid += nodes[element[i]-1,:3] 
        centroid /= 4.0
        v_pos = brain_invaff_mat.dot(np.append(centroid,1))
        label = brain_na[tuple(v_pos[:3].astype(int))]
        if label == 0 or label == 10 or label == 50:
            label = 40 # Set boundary elements to grey matter
            relabeled+=1
        ele_labels[idx] = label
        #print(idx+1,label)
    elif element[0] == 2:
        continue
    else:
        print("ERROR: Unexpected element type during element labeling.", element[0] )
        exit()        

if relabeled > 0:
    print('WARNING:', relabeled, 'elements did not have grey/white matter label and have been relabeled to grey matter')
    warnings+=1

elements=np.append(elements,ele_labels,axis=1)

### Output new mesh file (with labels)

labeled_msh_file=out_dir+name+'-labeled_brain.msh'
f_out = open(labeled_msh_file, 'w')
f_out.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
f_out.write(str(nodes_size)+'\n')
for idx, node in enumerate(nodes):
    f_out.write(str(idx+1)+' '+' '.join(map(str,node[:3]))+'\n')
f_out.write('$EndNodes\n$Elements\n')
f_out.write(str(elements_size)+'\n')
for idx, element in enumerate(elements):
    if element[0] == 2:
        f_out.write(str(idx+1)+' 2 2 2000 1 '+' '.join(map(str,element[1:4]))+'\n')
    elif element[0] == 4:
        f_out.write(str(idx+1)+' 4 2 '+str(element[5])+' 1 '+' '.join(map(str,element[1:5]))+'\n')
    else:
       print("ERROR: Unexpected element type during labeled mesh output.", element[0] )
       exit()
f_out.write('$EndElements\n')
f_out.close()

print("The labeled mesh file has been successfully output at", labeled_msh_file)
    
if warnings: print('There have been '+str(warnings)+' WARNINGS')
