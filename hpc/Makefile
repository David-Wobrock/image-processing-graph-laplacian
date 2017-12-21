PETSC_DIR=/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/wobrock/petsc-3.8.2
PETSC_ARCH=arch-linux2-c-debug
SLEPC_DIR=/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/wobrock/slepc-3.8.2

PETSC_CONFIGDIR=${PETSC_DIR}/lib/petsc
SLEPC_CONFIGDIR=${SLEPC_DIR}/lib/slepc

all: image_processing

#include ${PETSC_CONFIGDIR}/conf/variables
#include ${PETSC_CONFIGDIR}/conf/rules
include ${SLEPC_CONFIGDIR}/conf/slepc_variables
include ${SLEPC_CONFIGDIR}/conf/slepc_rules

COMPILER=@mpicc
COMP_FLAGS=-Wall
LINKER=@mpicc
LINK_FLAGS=-Wall
ECHO=@echo
SRCS=${wildcard *.c}
OBJS=${SRCS:.c=.o}

PETSC_INCLUDES=-I${PETSC_DIR}/include/ -I${PETSC_DIR}/${PETSC_ARCH}/include/
SLEPC_INCLUDES=-I${SLEPC_DIR}/include/ -I${SLEPC_DIR}/${PETSC_ARCH}/include/
ALL_INCLUDES=${PETSC_INCLUDES} ${SLEPC_INCLUDES}

PETSC_LINKS=-L${PETSC_DIR}/${PETSC_ARCH}/lib/
SLEPC_LINKS=-L${SLEPC_DIR}/${PETSC_ARCH}/lib/
ALL_LINKS=${PETSC_LINKS} ${SLEPC_LINKS}

CUSTOM_LIBS=-lpng -lm
ALL_LIBS=${PETSC_LIB} ${SLEPC_LIB} ${CUSTOM_LIBS}

# ** Targets **
image_processing: ${OBJS}
	${ECHO} "Linking $@"
	${LINKER} -o $@ ${LINK_FLAGS} ${OBJS} ${ALL_LINKS} ${ALL_LIBS}

%.o: %.c
	${ECHO} "Compiling $@"
	${COMPILER} -c ${COMP_FLAGS} ${@:.o=.c} -o $@ ${ALL_INCLUDES}
