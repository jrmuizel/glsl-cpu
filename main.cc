
#include "glsl.h"
using namespace glsl;
vec4 gl_FragCoord;
vec4 gl_FragColor;
vec4 gl_Position;
typedef uint32_t GLuint;
typedef int32_t GLboolean;
typedef float GLfloat;

typedef int32_t GLint;
typedef int32_t GLsizei;
typedef uint32_t GLenum;
typedef size_t GLsizeiptr;

sampler2D lookup_sampler(int texture) {
        return 0;
}

isampler2D lookup_isampler(int texture) {
        return 0;
}

sampler2DArray lookup_sampler_array(int texture) {
        return 0;
}

struct VertexAttrib {
        GLint size;
        GLenum type;
        bool normalized;
        GLsizei stride;
        GLuint offset;
        bool enabled = false;
        GLuint divisor;
        int vertex_array;
        char *buf; // XXX: this can easily dangle
};

#include "all.h"


int main() {
}
