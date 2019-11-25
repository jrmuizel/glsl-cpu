CXX=clang++
CXXFLAGS=-std=c++17 -O2 -g -ferror-limit=800
main: glsl.h main.cc shaders/all.h
	$(CXX) -DNDEBUG -Ishaders/ $(CXXFLAGS) main.cc -o main
	#$(CXX) -DNDEBUG -Is $(CXXFLAGS) main.cc -o main
shaders/all.h:
	cd shaders && ./parse
clean:
	rm main
