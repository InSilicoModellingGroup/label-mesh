import nibabel as nib
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

""" 
This script creates an array nx3 of nodes cordinates (n is number of
nodes). Then it determines the hounsfield unit, radiation value and tumour value
for each node and appends it as a new column of the nodes array.

> Hounsfield is determined from the CT image. Each pixel of the CT images is
  imported in a 3D array. Then using the inverse affine matrix we can convert
  world coordinates to pixel coordinates to extract the HU.

> Radiation field is obtained from a different image file in a similar way to
  Hounsfield. There are two such images corresponding to phase 1 and phase 2 of
  radiotherapy.

> Cancer field is zero everywhere for now.
"""


### Specify required files (CT images, Mesh)
nifti_ct_file = '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/01_Phase_I_plan/01_Phase_I_plan_20161010115024_1a.nii'
#'/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/08_Phase_I_plan/08_Phase_I_plan_20130111124359_1a.nii'
#'/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/07_Phase_I_plan/07_Phase_I_plan_20150320093218_1.nii'
#'/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/07_Phase_I_plan/07_Phase_I_plan_20160104125035_1a.nii'
#'/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/05_Phase_I_plan/05_Phase_I_plan_20150525082852_1a.nii'
nifti_rd_files = ['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/01_Phase_I_plan/01_Phase_I_plan_20161010115024_1.nii',
                  '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/01_Phase_II_plan/01_Phase_II_plan_20161010115024_1.nii']
#['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/08_Phase_I_plan/08_Phase_I_plan_20130111124359_1.nii',
#'/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/08_Phase_II_plan/08_Phase_II_plan_20130111124359_1.nii']
#['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/01_Phase_I_plan/01_Phase_I_plan_20161010115024_1.nii',
#                  '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/01_Phase_II_plan/01_Phase_II_plan_20161010115024_1.nii']
#['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/08_Phase_I_plan/08_Phase_I_plan_20130111124359_1.nii',
#                  '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/08_Phase_II_plan/08_Phase_II_plan_20130111124359_1.nii']
#['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/07_Phase_I_plan/07_Phase_I_plan_20150320093218_1a.nii',
#                  '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/07_Phase_II_plan/07_Phase_II_plan_20150320093218_1.nii']
#['/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/05_Phase_I_plan/05_Phase_I_plan_20150525082852_1.nii',
#                 '/home/schlang/pool/LungFibrosisData/BOCOC-LungFibrosis-Baseline/05_Phase_II_plan/05_Phase_II_plan_20150525082852_1.nii']
### First working case
# "/home/schlang/pool/LungFibrosisData/Prognosis-LungFibrosis/Export0001/DICOMIMG/SR0000/SR0000_P_Abd+Pelvis_C_Abdomen_Hx_0_105890.nii"
msh_file = '/home/schlang/Main/Study/PostDoc/InSilico/LungSim/data/LungSegmentations/seg_bococ_p1/seg_bococ_p1_ct0_right.msh'
### First working case
# "/home/schlang/Main/Study/PostDoc/InSilico/LungSim/data/LungSegmentations/seg_prog_case1/case1-right-new.msh"


### Case name (used in output files)
#name="case1-right"
name="p1_ct0_right"
### Specify output
out_dir='' #/home/schlang/pool/'

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

if nodes_size == nodes[:,0].size:
    print("Imported ", nodes[:,0].size, " nodes.")
else:
    print("ERROR: Number of imported nodes does not match nodes specified in msh file")

### Obtain inverse affine matrix

# Load ct images file
ct_img = nib.load(nifti_ct_file)

# Convert the voxel orientation to RAS
ct_img = nib.as_closest_canonical(ct_img)

print('[CT image] Voxel orientation is '+str(nib.aff2axcodes(ct_img.affine)))
ct_invaff_mat=np.linalg.inv(ct_img.affine)

# Get data from ct to a numpy array
ct_na = ct_img.get_fdata()
print('[CT image] Dimensions', ct_img.shape)

# Plot a slice (for testing purposes)
#slice=ct_na[:,:,66]
#plt.imshow(slice.T, cmap='gray', origin='lower')
#plt.show()

### Populate Hounsfield value for each node
hf=np.zeros((nodes_size,1))
for idx, node in enumerate(nodes):
    v_pos = ct_invaff_mat.dot(np.append(node[:3],1))
    hf[idx] = ct_na[tuple(v_pos[:3].astype(int))]
    #print(nodes[0,:],v_pos[:3].astype(int))

nodes=np.append(nodes,hf,axis=1)

### Populate radiation value for each node

img=0
# Load radiation dose images file
for nifti_rd_file in nifti_rd_files:

    rd_img = nib.load(nifti_rd_file)

    # Convert the voxel orientation to RAS
    rd_img = nib.as_closest_canonical(rd_img)

    print('[RD image '+str(img)+'] Voxel orientation is '+str(nib.aff2axcodes(rd_img.affine)))
    rd_aff_mat = rd_img.affine
    # IMPORTANT: When converting radiation dose dicom to nifty, slice thickness is not
    # transferred. We set it to 3 mm which is used in BOCOC images but this is not
    # applicaple everywhere
    if rd_aff_mat[2,2] == 1:
        rd_aff_mat[2,2] = 3
    rd_invaff_mat=np.linalg.inv(rd_aff_mat)

    # Get data from ct to a numpy array
    rd_na = rd_img.get_fdata()
    print('[RD image] Dimensions', rd_img.shape)

    rd=np.zeros((nodes_size,1))
    for idx, node in enumerate(nodes):
        v_pos = rd_invaff_mat.dot(np.append(node[:3],1))
        rd[idx] = rd_na[tuple(v_pos[:3].astype(int))]
        # IMPORTANT: This was found to be required to match Slicer data
        if img == 0:
            rd[idx] = rd[idx] / 197.49 #/ 198.5 (for p8)
        elif img == 1:
            rd[idx] = rd[idx] / 694.07 #/ 81.111 (for p8)
            #print(nodes[0,:],v_pos[:3].astype(int))
    print('Maximum radiation value:', np.max(rd))
    nodes=np.append(nodes,rd,axis=1)
    img=img+1

ARTIFICIAL_RA=False
if ARTIFICIAL_RA:
    #### Set position of mean depending on nodes coordinates range
    x_min=np.min(nodes[:,0])
    y_min=np.min(nodes[:,1])
    z_min=np.min(nodes[:,2])
    x_max=np.max(nodes[:,0])
    y_max=np.max(nodes[:,1])
    z_max=np.max(nodes[:,2])
    
    x_var=(x_max-x_min)
    y_var=(y_max-y_min)
    z_var=(z_max-z_min)
    x_mean=x_var/2 + x_min
    y_mean=y_var/4 + y_min
    z_mean=z_var/3 + z_min
    #print(x_max,y_max,z_max)

    x_var_coeff,y_var_coeff,z_var_coeff=1,0.5,0.2
    
    rd=np.zeros((nodes_size,1))
    for index, node in enumerate(nodes):
        rd[index]=math.exp(-((node[0]-x_mean)/(x_var*x_var_coeff))**2) * math.exp(-((node[1]-y_mean)/(y_var*y_var_coeff))**2) * math.exp(-((node[2]-z_mean)/(z_var*z_var_coeff))**2)
        rd[index]*=70

### Populate tumour value for each node
tm=np.zeros((nodes_size,1))
nodes=np.append(nodes,tm,axis=1)

### Output to file
nodal_field_file=out_dir+name+'-nodal_field.dat'
f_out = open(nodal_field_file, 'w')
f_out.write('4\n\n')
f_out.write('v02 a00 a01 v00\n\n')
for node in nodes:
    f_out.write(' '.join(map(str,node[3:]))+'\n')
    # output node coordinates as well
    #f_out.write(' '.join(map(str,node))+'\n')
f_out.close()

print("The nodal field has been successfully output at", nodal_field_file)
