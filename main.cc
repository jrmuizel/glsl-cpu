
#include "glsl.h"
using namespace glsl;
typedef uint32_t GLuint;
typedef int32_t GLboolean;
typedef float GLfloat;

typedef int32_t GLint;
typedef int32_t GLsizei;
typedef uint32_t GLenum;
typedef size_t GLsizeiptr;

sampler2D lookup_sampler(sampler2D_impl *s, int slot) {
        return 0;
}

isampler2D lookup_isampler(isampler2D_impl *s, int slot) {
        return 0;
}

sampler2DArray lookup_sampler_array(sampler2DArray_impl *s, int slot) {
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

struct ProgramImpl {
    virtual ~ProgramImpl() {}
    virtual const char *get_name() const = 0;
    virtual int get_uniform(const char *name) const = 0;
    virtual void bind_attrib(const char *name, int index) = 0;
    virtual void init_shaders(void *vertex_shader, void *fragment_shader) = 0;
};

struct ShaderImpl {
    typedef void (ShaderImpl::*SetUniform1iFunc)(int index, int value);
    typedef void (ShaderImpl::*SetUniform4fvFunc)(int index, const float *value);
    typedef void (ShaderImpl::*SetUniformMatrix4fvFunc)(int index, const float *value);

    SetUniform1iFunc set_uniform_1i_func = nullptr;
    SetUniform4fvFunc set_uniform_4fv_func = nullptr;
    SetUniformMatrix4fvFunc set_uniform_matrix4fv_func = nullptr;
};

struct VertexShaderImpl : ShaderImpl {
    typedef void (VertexShaderImpl::*InitBatchFunc)();
    typedef void (VertexShaderImpl::*LoadAttribsFunc)(VertexAttrib *attribs, unsigned short *indices, int start, int instance, int count);
    typedef void (VertexShaderImpl::*RunFunc)(char* flats, char* interps, size_t interp_stride);

    InitBatchFunc init_batch_func = nullptr;
    LoadAttribsFunc load_attribs_func = nullptr;
    RunFunc run_func = nullptr;

    vec4 gl_Position;
};

struct FragmentShaderImpl : ShaderImpl {
    typedef void (FragmentShaderImpl::*InitBatchFunc)();
    typedef void (FragmentShaderImpl::*InitPrimitiveFunc)(const void* flats);
    typedef void (FragmentShaderImpl::*InitSpanFunc)(const void* interps, const void* step);
    typedef void (FragmentShaderImpl::*RunFunc)(const void* step);
    typedef void (FragmentShaderImpl::*SkipFunc)(const void* step);
    typedef bool (FragmentShaderImpl::*UseDiscardFunc)();

    InitBatchFunc init_batch_func = nullptr;
    InitPrimitiveFunc init_primitive_func = nullptr;
    InitSpanFunc init_span_func = nullptr;
    RunFunc run_func = nullptr;
    SkipFunc skip_func = nullptr;
    UseDiscardFunc use_discard_func = nullptr;

    vec4 gl_FragCoord;
    Bool isPixelDiscarded;
    vec4 gl_FragColor;

    void step_fragcoord() {}
};

template<typename T>
void load_attrib(T& attrib, VertexAttrib &va, unsigned short *indices, int start, int instance, int count) {
}

#include "all.h"


int main() {
}
