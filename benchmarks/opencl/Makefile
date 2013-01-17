EXECS=vector-add

all=vector-add

vector-add: vector-add.c
	$(CC) -o vector-add vector-add.c -I/opt/cuda/include -lOpenCL -lm

clean:
	rm $(EXECS) *.o
