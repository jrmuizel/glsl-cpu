CXXFLAGS=-std=c++17 -O2 -g -ferror-limit=800
main: glsl.h brush_image.vs.h main.cc
	c++ -DNDEBUG -I../glsl-to-cpp/shaders/ $(CXXFLAGS) main.cc -o main
	#c++ -DNDEBUG -Is $(CXXFLAGS) main.cc -o main
clean:
	rm main
