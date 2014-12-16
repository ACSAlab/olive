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
force_x64 = 1
force_sm35 = 1
disable_l1_cache = 0


#-------------------------------------------------------------------------------
# Project Directories
#-------------------------------------------------------------------------------
PROJECT_ROOT = ..
SRC_DIR      = $(PROJECT_ROOT)/src
OLIVE_DIR    = $(SRC_DIR)/olive
UTEST_DIR    = $(SRC_DIR)/unitest
ALGO_DIR     = $(SRC_DIR)/algorithm
BUILD_DIR    = $(PROJECT_ROOT)/build
OBJ_DIR      = $(BUILD_DIR)/obj
BIN_DIR      = $(BUILD_DIR)/bin
LIB_DIR      = $(BUILD_DIR)/lib

#-------------------------------------------------------------------------------
# CUDA Directories
#-------------------------------------------------------------------------------
CUDA_INC_DIR = "$(shell dirname $(NVCC))/../include"
CUDA_LIB_DIR = "$(shell dirname $(NVCC))/../lib64"

#-------------------------------------------------------------------------------
# Compliers and flags
#-------------------------------------------------------------------------------
CC = g++
NVCC = "$(shell which nvcc)"
NVCCFLAGS = -G -g -std=c++11
CCFLAGS = -Wall -g -std=c++11

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
# 64 or 32 bit (compile with 32-bit device pointers by default)
#-------------------------------------------------------------------------------
ifneq ($(force_x64), 1)
	NVCCFLAGS += -m32
	CCFLAGS += -m32
else
	NVCCFLAGS += -m64
	CCFLAGS += -m64
endif

#-------------------------------------------------------------------------------
# Includes & Libraries
#-------------------------------------------------------------------------------
INC = -I$(CUDA_INC_DIR) -I$(OLIVE_DIR)
LIB = -L$(CUDA_LIB_DIR)
DLL = -lcudart

#-------------------------------------------------------------------------------
# Generates the name lists for building
#-------------------------------------------------------------------------------
CUSRC   = $(wildcard $(OLIVE_DIR)/*.cu)
CUOBJ   = $(patsubst $(OLIVE_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(CUSRC))
CCSRC   = $(wildcard $(OLIVE_DIR)/*.cc)
CUOBJ   = $(patsubst $(OLIVE_DIR)/%.cc, $(OBJ_DIR)/%.cc.o, $(CCSRC))
UTSRC   = $(wildcard $(UTEST_DIR)/*.cc)
UTEXE   = $(patsubst $(UTEST_DIR)/%.cc, $(BIN_DIR)/%, $(UTSRC))
UTCUSRC = $(wildcard $(UTEST_DIR)/*.cu)
UTCUEXE = $(patsubst $(UTEST_DIR)/%.cu, $(BIN_DIR)/%, $(UTCUSRC))
DEPS    = $(wildcard $(OLIVE_DIR)/*.h)


#-------------------------------------------------------------------------------
# Fake Targets
#-------------------------------------------------------------------------------
.PHONY: clean clean-bin clean-all libolive objects

all: 


#-------------------------------------------------------------------------------
# Archive olive static library
#-------------------------------------------------------------------------------
libolive: $(LIB_DIR) objects
	@echo "Creating library: " $@
	@ar -r -u $@ $(CUOBJS)


#-------------------------------------------------------------------------------
# Target/common rule for building olive codes
#-------------------------------------------------------------------------------
objects: $(OBJ_DIR) $(CUOBJ) $(CCOBJ)

$(CUOBJ): $(OBJ_DIR)/%.cu.o: $(OLIVE_DIR)/%.cu $(DEPS)
	$(NVCC) -o $@ $< $(NVCCFLAGS) $(INC) -c

$(CCOBJ): $(OBJ_DIR)/%.cc.o: $(OLIVE_DIR)/%.cc $(DEPS)
	$(CC) -o $@ $< $(CCFLAGS) $(INC) -c


#-------------------------------------------------------------------------------
# Rules for unitest (unitest/%.cc => bin/%)
#-------------------------------------------------------------------------------
unitests: $(BIN_DIR) $(UTEXE) $(UTCUEXE)

$(UTEXE): $(BIN_DIR)/%: $(UTEST_DIR)/%.cc $(DEPS) 
	$(CC) -o $@ $< $(CCFLAGS) $(INC) $(LIB) $(DLL)

$(UTCUEXE): $(BIN_DIR)/%: $(UTEST_DIR)/%.cu $(DEPS) 
	$(NVCC) -o $@ $< $(NVCCFLAGS) $(INC) $(LIB) $(DLL)

#-------------------------------------------------------------------------------
# Rules for BFS
#-------------------------------------------------------------------------------
#bfs: $(BIN_DIR) $(BIN_DIR)/test_bfs

#$(BIN_DIR)/test_bfs: $(TEST_DIR)/test_bfs.cu libs
#	@echo "Building executable: "$@	
#	@$(NVCC) -o $@ $< $(NVCCFLAGS) $(INC) $(LIB_DIR)/libolive.a



#-------------------------------------------------------------------------------
# Rules for build directories
#-------------------------------------------------------------------------------
$(BIN_DIR): $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR): $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)

$(LIB_DIR): $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)


#-------------------------------------------------------------------------------
# Cleaning up
#-------------------------------------------------------------------------------
clean:
	rm -f $(OBJ_DIR)/*.o 

clean-bin: 
	rm -rf $(BIN_DIR)/*

clean-all: 
	rm -rf $(BUILD_DIR)/*

