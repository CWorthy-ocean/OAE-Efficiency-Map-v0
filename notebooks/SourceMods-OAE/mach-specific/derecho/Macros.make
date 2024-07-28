SUPPORTS_CXX := FALSE
ifeq ($(COMPILER),intel)
  FFLAGS :=  -qno-opt-dynamic-align  -convert big_endian -assume byterecl -ftz -traceback -assume realloc_lhs -fp-model source  
  SUPPORTS_CXX := TRUE
  CXX_LINKER := FORTRAN
  FC_AUTO_R8 :=  -r8 
  FFLAGS_NOOPT :=  -O0 
  FIXEDFLAGS :=  -fixed  
  FREEFLAGS :=  -free 
endif
MPICC :=  cc 
MPICXX :=  CC 
MPIFC :=  ftn 
SCC :=  cc 
SCXX :=  CC 
SFC :=  ftn 
CMAKE_OPTS :=  -DCMAKE_SYSTEM_NAME=Catamount
PIO_FILESYSTEM_HINTS := lustre
CFLAGS :=   -qno-opt-dynamic-align -fp-model precise -std=gnu99 
NETCDF_PATH := $(NETCDF)
PNETCDF_PATH := $(PNETCDF)
CPPDEFS := $(CPPDEFS)  -DCESMCOUPLED 
CPPDEFS := $(CPPDEFS)  -DLINUX 
SLIBS := $(SLIBS)  -lnetcdff -lnetcdf 
ifeq ($(MODEL),pop)
  CPPDEFS := $(CPPDEFS)  -D_USE_FLOW_CONTROL 
endif
ifeq ($(MODEL),ufsatm)
  CPPDEFS := $(CPPDEFS)  -DSPMD 
  FFLAGS := $(FFLAGS)  $(FC_AUTO_R8) 
endif
ifeq ($(MODEL),gptl)
  CPPDEFS := $(CPPDEFS)  -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY  
endif
ifeq ($(MODEL),mom)
  FFLAGS := $(FFLAGS)  $(FC_AUTO_R8) -Duse_LARGEFILE
endif
ifeq ($(MODEL),mpi-serial)
  CFLAGS := $(CFLAGS)  -std=gnu89 
endif
ifeq ($(COMPILER),intel)
  FFLAGS := $(FFLAGS)  -march=core-avx2 -no-fma
  CPPDEFS := $(CPPDEFS)  -DFORTRANUNDERSCORE -DCPRINTEL
  CFLAGS := $(CFLAGS)  -march=core-avx2 -no-fma
  ifeq ($(compile_threaded),TRUE)
    FFLAGS := $(FFLAGS)  -qopenmp 
    CFLAGS := $(CFLAGS)  -qopenmp 
  endif
  ifeq ($(DEBUG),TRUE)
    FFLAGS := $(FFLAGS)  -O0 -g -check uninit -check bounds -check pointers -fpe0 -check noarg_temp_created 
    CFLAGS := $(CFLAGS)  -O0 -g 
  endif
  ifeq ($(DEBUG),FALSE)
    FFLAGS := $(FFLAGS)  -O2 -debug minimal 
    CFLAGS := $(CFLAGS)  -O2 -debug minimal 
  endif
  ifeq ($(MPILIB),mpich)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),mpich2)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),mvapich)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),mvapich2)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),mpt)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),openmpi)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),impi)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(MPILIB),mpi-serial)
    SLIBS := $(SLIBS)  -mkl 
  endif
  ifeq ($(compile_threaded),TRUE)
    LDFLAGS := $(LDFLAGS)  -qopenmp 
  endif
endif
ifeq ($(MODEL),ufsatm)
  INCLDIR := $(INCLDIR)  -I$(EXEROOT)/atm/obj/FMS 
endif
