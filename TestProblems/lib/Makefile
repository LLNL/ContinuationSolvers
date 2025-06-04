# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = mpicxx
F90       = mpifort
HYPRE_DIR = /g/g20/tuckerh/software/hypre-2.25.0/src/hypre#/home/tucker/software/glvis_mfem/hypre/src/hypre


########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = 
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)


LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c

.c.o:
	$(CC) $(CFLAGS) -c $<

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = diablo

all: $(ALLPROGS)

default: all

########################################################################
# diablolib
########################################################################
diablo: diablo.o
	$(CC) -o $@ $^ $(LFLAGS)

diablolib.a: diablo.o # create a static library
	ar rcs diablolib.a diablo.o

scratch: scratch.o
	$(CC) -o $@ $^ $(LFLAGS)
 
########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o) diablolib.a
