CXXFLAGS=-std=c++17 -O2 -g -ferror-limit=800
main: glsl.h brush_image.vs.h main.cc shaders/all.h
	c++ -DNDEBUG -Ishaders/ $(CXXFLAGS) main.cc -o main
	#c++ -DNDEBUG -Is $(CXXFLAGS) main.cc -o main
shaders/all.h:
	cd shaders && ./parse
clean:
	rm main
