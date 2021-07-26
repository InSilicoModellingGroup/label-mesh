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

> Hounsfield is determined from the CT scan. Each pixel of the CT images is
  imported in a 3D array. Then using the inverse affine matrix we can convert
  world coordinates to pixel coordinates to extract the HU.
> Radiation field is actually a mock case calculated using spherical normal
  distributions centered at a specified point in the lung.
> Cancer field is zero everywhere for now.
"""


### Specify required files (CT images, Mesh)
nifti_file = '/home/schlang/Downloads/Fibrosis_Cases1-7/Nicos/Export0000/DICOMIMG/SR0000/SR0000_P_Abd+Pelvis_C_Abdomen_Hx_0_105890.nii'
#msh_file = '/home/schlang/pool/lung-segmentation/Segmentation_Left-lung-new-optnetgen.msh'
msh_file = '/home/schlang/pool/lung-segmentation/case1-right-new.msh'

### Case name (used in output files)
#name="case1-left"
name="case1-right"
### Specify output
out_dir='/home/schlang/pool/lung-segmentation/'

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
ct_img = nib.load(nifti_file)

# Convert the voxel orientation to RAS
ct_img = nib.as_closest_canonical(ct_img)

print('Voxel orientation is '+str(nib.aff2axcodes(ct_img.affine)))
invaff_mat=np.linalg.inv(ct_img.affine)

# Get data from ct to a numpy array
ct_na = ct_img.get_fdata(dtype=np.float32)
print('CT image dimensions', ct_img.shape)

# Plot a slice (for testing purposes)
#slice=ct_na[:,:,66]
#plt.imshow(slice.T, cmap='gray', origin='lower')
#plt.show()

### Populate Hounsfield value for each node
hf=np.zeros((nodes_size,1))
for idx, node in enumerate(nodes):
    v_pos = invaff_mat.dot(np.append(node,1))
    hf[idx] = ct_na[tuple(v_pos[:3].astype(int))]
    #print(nodes[0,:],v_pos[:3].astype(int))

nodes=np.append(nodes,hf,axis=1)

### Populate radiation value for each node

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

ra=np.zeros((nodes_size,1))
for index, node in enumerate(nodes):
    ra[index]=math.exp(-((node[0]-x_mean)/(x_var*x_var_coeff))**2) * math.exp(-((node[1]-y_mean)/(y_var*y_var_coeff))**2) * math.exp(-((node[2]-z_mean)/(z_var*z_var_coeff))**2)
    ra[index]*=70

#print(np.max(ra))
nodes=np.append(nodes,ra,axis=1)

### Populate tumour value for each node
tm=np.zeros((nodes_size,1))
nodes=np.append(nodes,tm,axis=1)

### Output to file
nodal_field_file=out_dir+name+'-nodal_field.dat'
f_out = open(nodal_field_file, 'w')
f_out.write('3\n\n')
f_out.write('v02 a00 v00\n\n')
for node in nodes:
    f_out.write(' '.join(map(str,node[3:]))+'\n')
    # output node coordinates as well
    #f_out.write(' '.join(map(str,node))+'\n')
f_out.close()

print("The nodal field has been successfully output at", nodal_field_file)
