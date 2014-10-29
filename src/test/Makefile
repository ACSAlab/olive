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
# Compiler switches
#-------------------------------------------------------------------------------
force_cuda5 = 1
force64 = 1
force_sm35 = 0
disable_l1_cache = 0


#-------------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------------
PROJECT_ROOT = ../..
OLIVE_DIR = $(PROJECT_ROOT)/src/olive
BUILD_DIR = $(PROJECT_ROOT)/build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin


#-------------------------------------------------------------------------------
# Compliers and flags
#-------------------------------------------------------------------------------
CC = gcc
CXX = g++
NVCCFLAGS = -O3
CCFLAGS = -Wall -O3


#-------------------------------------------------------------------------------
# NVCC version
#-------------------------------------------------------------------------------
ifneq ($(force_cuda5), 1)
	NVCC = "$(shell which nvcc)"
else
	NVCC = /usr/local/cuda-5.5/bin/nvcc
endif


#-------------------------------------------------------------------------------
# CUDA Capability 2.0 or 3.5
#-------------------------------------------------------------------------------
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"

ifneq ($(force_sm35), 1)
	NVCCFLAGS += $(GEN_SM20)
else
	NVCCFLAGS += $(GEN_SM35)
endif


#-------------------------------------------------------------------------------
# Disable L1 cache by "-Xptxas -dlcm=cg" option at compile time can 
# reduce over-fetch (e.g, in the case of scattered memory accesses)
#-------------------------------------------------------------------------------
ifneq ($(disable_l1_cache), 1)
	NVCCFLAGS += 
else
	NVCCFLAGS += -Xptxas -dlcm=cg
endif

#-------------------------------------------------------------------------------
# 64 or 32 bit
#-------------------------------------------------------------------------------
ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	NVCCFLAGS += -m32
	CCFLAGS += -m32
else
	NVCCFLAGS += -m64
	CCFLAGS += -m64
endif


#-------------------------------------------------------------------------------
# Includes (src and include bothin in OLIVE_DIR)
#-------------------------------------------------------------------------------
INC = -I$(OLIVE_DIR)



#-------------------------------------------------------------------------------
# Generates the name lists for building
#-------------------------------------------------------------------------------
CUSRC   = $(wildcard $(OLIVE_DIR)/*.cu)  
CUOBJS  = $(patsubst $(OLIVE_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(CUSRC)) 
DEPS = $(wildcard $(OLIVE_DIR)/*.h)


#-------------------------------------------------------------------------------
# Target/common rule for building olive codes
# Use make objects to build the objects only
#-------------------------------------------------------------------------------
objects: $(OBJ_DIR) $(CUOBJS)

$(OBJ_DIR)/%.cu.o: $(OLIVE_DIR)/%.cu $(DEPS)
	@echo "Creating cuda object: "$@
	@$(NVCC) -o $@ $< $(NVCCFLAGS) $(INC) -c


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------
.PHONY: clean clean-bin clean-all bfs


#-------------------------------------------------------------------------------
# Rules for BFS
#-------------------------------------------------------------------------------
bfs: $(BIN_DIR) $(BIN_DIR)/test_bfs

$(BIN_DIR)/test_bfs: test_bfs.cu objects $(DEPS)
	@printf "Building bfs...\n"
	@$(NVCC) -o $@ $< $(NVCCFLAGS) $(INC)


#-------------------------------------------------------------------------------
# Rules for build directories
#-------------------------------------------------------------------------------
$(BIN_DIR): $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR): $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------
clean:
	rm -f $(OBJ_DIR)/*.o 

clean-bin: 
	rm -rf $(BIN_DIR)/*

clean-all: 
	rm -rf $(BUILD_DIR)/*

