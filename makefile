#Compilers
PCC = icc


#Basic flags for compiling
FLAGS = -std=gnu99 -D_FILE_OFFSET_BITS=64

#Flags for performance
OPTFLAGS = -O3 -march=nocona

#Flags for debugging
#DBFLAGS = -g
#DBFLAGS = -g -Wunused-parameter -Wall -Wextra #-Wpadded

#Locations
LIBDIR = ./libs
OBJDIR = ./objects
BINDIR = ./bin

#Header and libraries
ASYNCH_HEADERS = -I/Users/apensoneault/asynch_python
ASYNCH_LIBSLOC = -L/Users/apensoneault/asynch_python/libs/ -Wl,-rpath=/Users/apensoneault/asynch_python/libs/
ASYNCH_LIBS = -lasynch
LIBS = -lm -lpq -lz 

#Objects
SHAREDOBJS = $(addprefix $(OBJDIR)/,rkmethods.o problems.o mathmethods.o riversys.o sort.o comm.o system.o processdata.o partition.o definetype.o misc.o rainfall.o solvers.o io.o \
	forcings.o compression.o date_manip.o asynch_interface.o modeloutputs.o data_types.o)
LIBPYOBJS = $(addprefix $(OBJDIR)/,asynch_interface_py.o)
ASYNCHDISTOBJS = $(addprefix $(OBJDIR)/,asynchdist.o)
ASYNCHCUSTOMOBJS = $(addprefix $(OBJDIR)/,asynchdist_custom.o)


#How to compile and link
$(OBJDIR)/%.o: %.c
	$(PCC) -c $*.c $(HEADERS) $(FLAGS) $(DBFLAGS) $(OPTFLAGS) $(EXTRA_FLAGS) -o $(OBJDIR)/$*.o

ASYNCHLIB:
	$(MAKE) ASYNCHLIB_TARGET EXTRA_FLAGS=-fPIC

ASYNCHLIB_TARGET: $(SHAREDOBJS)
	$(PCC) $(SHAREDOBJS) $(LIBS) $(FLAGS) $(DBFLAGS) $(OPTFLAGS) -shared -Wl,-soname,libasynch.so -o $(LIBDIR)/libasynch.so

ASYNCHLIB_PY:
	$(MAKE) ASYNCHLIB_PY_TARGET EXTRA_FLAGS="-fPIC -I/usr/include/python2.7/"

ASYNCHLIB_PY_TARGET: $(SHAREDOBJS) $(LIBPYOBJS)
	$(PCC) $(SHAREDOBJS) $(LIBPYOBJS) $(LIBS) $(EXTRA_FLAGS) -L/usr/lib/python2.7/ -lpython2.7 $(FLAGS) $(DBFLAGS) $(OPTFLAGS) -shared -Wl,-soname,libasynch_py.so -o $(LIBDIR)/libasynch_py.so

ASYNCH: $(ASYNCHDISTOBJS)
	$(PCC) $(ASYNCHDISTOBJS) $(ASYNCH_HEADERS) $(ASYNCH_LIBSLOC) $(ASYNCH_LIBS) $(LIBS) $(FLAGS) $(DBFLAGS) $(OPTFLAGS) -o $(BINDIR)/ASYNCH

ASYNCHCUSTOM: $(ASYNCHCUSTOMOBJS)
	$(PCC) $(ASYNCHCUSTOMOBJS) $(ASYNCH_HEADERS) $(ASYNCH_LIBSLOC) $(ASYNCH_LIBS) $(LIBS) $(FLAGS) $(DBFLAGS) $(OPTFLAGS) -o $(BINDIR)/ASYNCHCUSTOM


clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(LIBDIR)/*.so
	rm -f $(BINDIR)/*
