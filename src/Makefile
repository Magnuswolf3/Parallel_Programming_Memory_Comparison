INC="./inc"
FLAGS = -I$(INC)
OMPFLAG = -fopenmp
NVCC = nvcc
LDFLAGS = -I. -lm 
CFLAGS = -DNDEBUG
CC = g++
OBJFILE0 = Serial/q_1.cu 
OBJFILE1 = Global/q_2.cu
OBJFILE2 = Shared/q_3.cu
OBJFILE3 = Texture/q_4.cu
TARGET0 = Serial/q1
TARGET1 = Global/q2
TARGET2 = Shared/q3
TARGET3 = Texture/q4

all: Serial/q1 Global/q2 Shared/q3 Texture/q4

Serial/q1: Serial/q_1.cu 
	$(NVCC) $(OBJFILE0) -o $(TARGET0)

Global/q2: Global/q_2.cu
	$(NVCC) $(OBJFILE1) -o $(TARGET1)

Shared/q3: Shared/q_3.cu
	$(NVCC) $(OBJFILE2) -o $(TARGET2)

Texture/q4: Texture/q_4.cu
	$(NVCC) $(OBJFILE3) -o $(TARGET3)

clean:
	rm -f $(TARGET0) $(TARGET1) $(TARGET2) $(TARGET3) *~
