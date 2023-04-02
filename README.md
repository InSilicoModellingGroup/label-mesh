### Description

Script for populating values for the nodes of a computational mesh (in .msh
format) using data from medical images (in nifti format). It is able to fill-in
Hounsfield units and radiation distribution from separate medical images.

More details about how the script works can be found in the .py file.

### Usage

In the .py file, the user needs to specify

* Path of folder with data (`basepath`)
* Names of nifti images (`nifti_*_file`)
* Name of msh file (`msh_file`)

Then run `python nodalfield-msh.py`

### Example

Data for one case are included in directory `p01`

After running the script, the output can be verified using the reference file
`p01_ct0_right-nodal_field.dat.verify`

