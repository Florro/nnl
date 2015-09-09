# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include config.mk
include /home/niklas/packages/mshadow/make/mshadow.mk
export CFLAGS = -Wall -O3 -I/home/niklas/packages/mshadow -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS= -lm $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS) -lcudnn -lopencv_highgui -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lboost_system

# specify tensor path
BIN =
OBJ =
CUOBJ =
CUBIN = dist_conv_NNL
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~

dist_conv_NNL: dist_conv_NNL.cu
