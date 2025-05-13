.PHONY: clean

L1TSC82ProngJetModel_v0.so: L1TSC82ProngJetModel_emulator_v0.o L1TSC82ProngJetModel_v0.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm L1TSC82ProngJetModel_emulator_v0.o L1TSC82ProngJetMode_v0.o L1TSC82ProngJetModel_v0.so
