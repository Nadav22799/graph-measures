# target: dependencies
# <TAB> linux commands

# $@ is the name of the current target 
# $< is used to sign the cpp file for the object target?


SO_FLAGS = -shared -fPIC
OBJ_FLAGS = -std=c++11 -fPIC -c
CC = g++

MOTIF_FILES = CacheGraph.o MotifCalculator.o MotifUtils.o MotifVariationConstants.o motifs.o

all: motif.so a.out order

# a.out: CacheGraph.o MotifCalculator.o MotifUtils.o MotifVariationConstants.o motifs.o
# $(CC) CacheGraph.o MotifCalculator.o MotifUtils.o MotifVariationConstants.o motifs.o -std=c++11

# real targets
a.out: $(MOTIF_FILES)
	$(CC) $(MOTIF_FILES) -std=c++11

motif.so: CacheGraph.o MotifCalculator.o MotifUtils.o MotifVariationConstants.o motifs.o
	$(CC) -o motif.so $(SO_FLAGS) CacheGraph.o MotifCalculator.o MotifUtils.o MotifVariationConstants.o motifs.o -std=c++11	

# This line converts all cpp files into obj files:
.cpp.o:
	$(CC) $(OBJ_FLAGS) $< -o $@

order:
	mkdir -p bin
	rm *.o
	mv *.so a.out bin

clean:
	rm *.o *.so a.out
