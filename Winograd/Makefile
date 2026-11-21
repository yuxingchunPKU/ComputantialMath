CXX      ?= g++
OPT      ?= -O2
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic $(OPT)
CPPFLAGS ?= -Iinclude
LDFLAGS  ?=

SRCS    := src/winograd.cpp src/linear.cpp src/bench.cpp
OBJS    := $(SRCS:.cpp=.o)
HDRS    := $(wildcard include/*.hpp)
TARGET  := bench

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

src/%.o: src/%.cpp $(HDRS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
