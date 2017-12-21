PETSC_DIR=/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/wobrock/petsc-3.8.2
PETSC_ARCH=arch-linux2-c-debug
SLEPC_DIR=/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/wobrock/slepc-3.8.2

PETSC_CONFIGDIR=${PETSC_DIR}/lib/petsc
SLEPC_CONFIGDIR=${SLEPC_DIR}/lib/slepc

all: image_processing

include ${PETSC_CONFIGDIR}/conf/variables
include ${PETSC_CONFIGDIR}/conf/rules
include ${SLEPC_CONFIGDIR}/conf/slepc_variables
include ${SLEPC_CONFIGDIR}/conf/slepc_rules

COMPILER=mpicc
PETSC_INCLUDES=-I${PETSC_DIR}/include/ -I${PETSC_DIR}/${PETSC_ARCH}/include/
PETSC_LINKS=-L${PETSC_DIR}/${PETSC_ARCH}/lib/ 
SLEPC_INCLUDES=-I${SLEPC_DIR}/include/ -I${SLEPC_DIR}/${PETSC_ARCH}/include/
SLEPC_LINKS=-L${SLEPC_DIR}/${PETSC_ARCH}/lib/ 

image_processing: image_processing.c read_img.c utils.c inverse_power_it.c gram_schmidt.c
	${COMPILER} image_processing.c read_img.c utils.c inverse_power_it.c gram_schmidt.c -Wall ${PETSC_INCLUDES} ${SLEPC_INCLUDES} ${PETSC_LINKS} ${SLEPC_LINKS} -o image_processing ${PETSC_LIB} ${SLEPC_LIB} -lpng -lm
