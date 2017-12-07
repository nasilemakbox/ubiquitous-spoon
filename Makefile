SOURCES = main.c d3tensor.c c3tensor.c decomp.c transpose.c fft.c ghostz.c ghost.c get.c io.c rk3.c fourth.c pressure.c diffuseu.c diffusew.c advect.c scalar.c tridag.c heptdag.c cases.c channel.c chanfast.c
OBJECTS := $(SOURCES:%.c=%.o)
TARGET = a.out
MPICC = mpicc
CFLAGS = -Wall -Wextra -O3 -funroll-loops
INCLUDES = -I$(HOME)/local/fftw-3.3.4/include
LIBS = -L$(HOME)/local/fftw-3.3.4/lib -lfftw3_mpi -lfftw3 -lm

main: $(OBJECTS)
	$(MPICC) $(CFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

clean:
	/bin/rm -f $(OBJECTS)

clobber: clean
	/bin/rm -f $(TARGET)

depend:
	makedepend -- $(CFLAGS) -- -Y $(SOURCES)

.c.o:
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

# DO NOT DELETE

main.o: check.h cases.h channel.h chanfast.h
d3tensor.o: check.h d3tensor.h
c3tensor.o: check.h c3tensor.h
decomp.o: check.h decomp.h
transpose.o: check.h transpose.h
fft.o: check.h d3tensor.h c3tensor.h fft.h
ghostz.o: check.h ghostz.h
ghost.o: check.h ghost.h
get.o: check.h get.h
io.o: check.h io.h
rk3.o: rk3.h
fourth.o: fourth.h
pressure.o: check.h tridag.h heptdag.h fourth.h pressure.h
diffuseu.o: check.h tridag.h heptdag.h fourth.h diffuseu.h
diffusew.o: check.h tridag.h heptdag.h fourth.h diffusew.h
advect.o: check.h fourth.h advect.h
scalar.o: check.h fourth.h scalar.h
tridag.o: check.h tridag.h
heptdag.o: check.h heptdag.h
cases.o: check.h d3tensor.h c3tensor.h decomp.h transpose.h fft.h ghostz.h
cases.o: ghost.h get.h io.h rk3.h pressure.h diffuseu.h diffusew.h advect.h
cases.o: cases.h
channel.o: check.h d3tensor.h c3tensor.h decomp.h transpose.h fft.h ghostz.h
channel.o: ghost.h get.h io.h rk3.h fourth.h pressure.h diffuseu.h diffusew.h
channel.o: advect.h scalar.h channel.h
chanfast.o: check.h d3tensor.h c3tensor.h decomp.h transpose.h fft.h ghostz.h
chanfast.o: ghost.h get.h io.h rk3.h fourth.h pressure.h diffuseu.h
chanfast.o: diffusew.h advect.h scalar.h chanfast.h
