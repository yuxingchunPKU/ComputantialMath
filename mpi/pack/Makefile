src = $(wildcard *.cpp)
obj = $(patsubst %.cpp,%.o,$(src))
args = -std=c++11 -O2
CXX=mpic++
all: main

main: $(obj)
	$(CXX) $^ -o $@ $(args) 

%.o: %.cpp
	$(CXX) -c $< -o $@ $(args) 

clean:
	rm -f main $(obj)

#目标: 依赖
#_tab_ 命令行1
#_tab_ 命令行2
# ... 


