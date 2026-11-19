#CC= /opt/local/bin/gcc-mp-13

all: fields.so reflections.so greenfct.so wavenumber.so

fields.so: fields.o
	gcc -shared -o $@ fields.o -lm

reflections.so: reflections.o
	gcc -shared -o $@ reflections.o -lm

greenfct.so: greenfct.o reflections.o fields.o
	gcc -shared -o $@ $^ -lm

wavenumber.so: wavenumber.o greenfct.o reflections.o fields.o
	gcc -shared -o $@ $^ -lm

#CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -fopt-info-optimized  -fopt-info-vec-missed  -march=znver2
#-fopt-info-optimized -fopenmp
CFLAGS= -fPIC -Ofast -ftree-loop-vectorize -ffast-math -fopt-info-optimized -funroll-all-loops 
CFLAGS= -fPIC -O3 -ftree-loop-vectorize -funroll-all-loops 
#CFLAGS= -fPIC -O0 -g

.SUFFIXES : .o .c 
.c.o    :
	$(CC) -c $(CFLAGS) $(OPTC) $<

clean:
	rm -f *.o

deepclean:
	rm -rf *.o *.so __pycache__

