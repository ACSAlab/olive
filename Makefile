#===============================================================================
# Test Makefile 
#
#
# Author: Yichao Cheng (onesuperclark@gmail.com)
# Created on: 2014-10-28
# Last Modified: 2014-10-28
#
#===============================================================================



#-------------------------------------------------------------------------------
# Compilers and flags
#-------------------------------------------------------------------------------
NVCC 		 = "$(shell which nvcc)"
NVCCFLAGS    = -G -g -std=c++11

force_x64 = 1
force_sm35 = 1
disable_l1_cache = 0

# CUDA Capability 2.0 or 3.5
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
ifneq ($(force_sm35), 1)
	NVCCFLAGS += $(GEN_SM20)
else
	NVCCFLAGS += $(GEN_SM35)
endif

# Disable L1 cache by "-Xptxas -dlcm=cg" option at compile time can 
# reduce over-fetch (e.g, in the case of scattered memory accesses)
ifneq ($(disable_l1_cache), 1)
	NVCCFLAGS += 
else
	NVCCFLAGS += -Xptxas -dlcm=cg
endif

# 64 or 32 bit (compile with 32-bit device pointers by default)
ifneq ($(force_x64), 1)
	NVCCFLAGS += -m32
	CCFLAGS += -m32
else
	NVCCFLAGS += -m64
	CCFLAGS += -m64
endif


#-------------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------------
CUDA_INC_DIR = "$(shell dirname $(NVCC))/../include"
CUDA_LIB_DIR = "$(shell dirname $(NVCC))/../lib64"

#-------------------------------------------------------------------------------
# Rules
#-------------------------------------------------------------------------------
OLIVE = $(wildcard *.h)

ALL = BFS 

TEST = testCsrGraph testBFS

all: $(ALL) $(TEST)

%: %.cu $(OLIVE)
	$(NVCC) -o $@ $< $(NVCCFLAGS) -I$(CUDA_INC_DIR) -L$(CUDA_LIB_DIR) -lcudart

.PHONY: clean

clean:
	rm -f $(OBJ_DIR)/*.o 


