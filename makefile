main.o : main.cpp
	@echo 'Building file: main.cpp'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -c -I/usr/local/include/ -fmessage-length=0 -std=c++11 -MMD -MP -MF"main.d" -MT"main.d" -o "main.o" "main.cpp"
	@echo 'Finished building: main.cpp'
	@echo ' '

all : decipher

decipher : main.o
	@echo 'Building target: main.o'
	@echo 'Invoking linker'
	g++ -L/usr/local/lib/ -o "decipher" "main.o" -lfst -lngram
	@echo 'Finished building target'
	@echo ' '

clean :
	rm "main.d" "main.o"