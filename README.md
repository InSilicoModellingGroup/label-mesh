### Description

Script for populating values for the nodes and labelling the elements of a computational mesh (in .msh
format) using data from medical images (in nifti format). This implementation is
particular to the PROTEAS project for brain metastatic cancer. It populates the
initial values for all variables of the model in one output file as well as Hounsfield and
radiation dosage units in a separate output file. Last, it also labels the
elements of the given mesh depending on whether they belong to while or grey
matter regions and outputs a labeled version of the mesh file.

More details about how the script works can be found in the .py file.

### Usage

In the .py file, the user needs to specify

* Path of folder with patient data (`basepath`)
* Names of nifti images (`nifti_*_file`)
* Name of msh file (`msh_file`)

Then run `python label-mesh.py`

### Example

Data for one case are included in directory `RP02`

After running the script, the output can be verified using the reference files
`*.verify`

