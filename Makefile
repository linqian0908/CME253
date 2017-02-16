#
#  Copyright 2016 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

ARCH=-arch sm_30
CUB_INCLUDE=../../../../cub-1.4.1
NVCC=nvcc

#gcc fdtd_cpu.c -o -lm fdtd_cpu

x.fdtd_sync: fdtd_sync.o
	$(NVCC) $(ARCH) -o x.fdtd_sync fdtd_sync.o

fdtd_sync.o: fdtd_sync.cu
	$(NVCC) $(ARCH) -c fdtd_sync.cu 
	
#clean:
#	rm -rf fdtd_cpu.o x.fdtd_cpu
