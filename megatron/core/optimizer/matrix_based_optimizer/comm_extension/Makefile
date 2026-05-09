PYTHON ?= python
CXX ?= c++
SRC ?= coalesced_collectives.cpp
OUT ?= _coalesced_collectives.so
COMM_CXXFLAGS ?=
COMM_LDFLAGS ?=

.PHONY: all
all:
	@mkdir -p "$$(dirname "$(OUT)")"
	$(CXX) $(COMM_CXXFLAGS) "$(SRC)" $(COMM_LDFLAGS) -o "$(OUT)"
