#include <stdint.h>
#include <xmmintrin.h>

// Every function in this file should be marked static and inline using SI.
#if defined(__clang__)
    #define SI __attribute__((always_inline)) static inline
#else
    #define SI static inline
#endif

namespace glsl {
template <typename T> using V = T __attribute__((ext_vector_type(4)));
using Float   = V<float   >;
using I32 = V< int32_t>;
using U64 = V<uint64_t>;
using U32 = V<uint32_t>;
using U16 = V<uint16_t>;
using U8  = V<uint8_t >;



SI Float if_then_else(I32 c, Float t, Float e) {
    return _mm_or_ps(_mm_and_ps(c, t), _mm_andnot_ps(c, e));
}

SI Float   min(Float a, Float b)       { return _mm_min_ps(a,b);    }
SI Float   max(Float a, Float b)       { return _mm_max_ps(a,b);    }

SI Float clamp(Float a, Float minVal, Float maxVal) {
        return min(max(a, minVal), maxVal);
}

SI Float sqrt(Float x) {
        return _mm_sqrt_ps(x);
}


SI Float step(Float edge, Float x) {
        return if_then_else(x < edge, 0, 1);
}



template <typename T>
SI T mix(T x, T y, Float a) {
        return (x - y) * a + x;
}

/*
enum RGBA {
        R,
        G,
        B,
        A
};*/

enum XYZW {
        X = 0,
        Y = 1,
        Z = 2,
        W = 3,
        R = 0,
        G = 1,
        B = 2,
        A = 3,
};


struct vec2 {
        vec2() { vec2(0); }
        vec2(Float a): x(a), y(a) {}
        vec2(Float x, Float y): x(x), y(y) {}
        Float x;
        Float y;

        vec2 operator*=(Float a) {
                x *= a;
                y *= a;
                return *this;
        }

        friend vec2 operator*(vec2 a, Float b) {
                return vec2(a.x*b, a.y*b);
        }

        friend vec2 operator-(vec2 a, vec2 b) {
                return vec2(a.x-b.x, a.y-b.y);
        }

};

vec2 step(vec2 edge, vec2 x) {
       return vec2(step(edge.x, x.x), step(edge.y, x.y));
}

vec2 max(vec2 a, vec2 b) {
       return vec2(max(a.x, b.x), max(a.y, b.y));
}

Float length(vec2 a) {
       return sqrt(a.x*a.x+a.y*a.y);
}
Float abs(Float v) {
        return _mm_and_ps(v, 0-v);
}

template <typename T, typename P>
T unaligned_load(const P* p) {  // const void* would work too, but const P* helps ARMv7 codegen.
    T v;
    memcpy(&v, p, sizeof(v));
    return v;
}

template <typename Dst, typename Src>
Dst bit_cast(const Src& src) {
    static_assert(sizeof(Dst) == sizeof(Src), "");
    return unaligned_load<Dst>(&src);
}

Float   cast  (U32 v) { return      __builtin_convertvector((I32)v,   Float); }

    Float floor(Float v) {
    #if defined(JUMPER_IS_SSE41)
        return _mm_floor_ps(v);
    #else
        Float roundtrip = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
        return roundtrip - if_then_else(roundtrip > v, 1, 0);
    #endif
    }

    U32 round(Float v, Float scale) { return _mm_cvtps_epi32(v*scale); }

Float fract(Float v) { return v - floor(v); }

Float fwidth(Float p) {
        return abs(p.yyww - p.xxzz) - abs(p.zwzw - p.xyxy);
}

vec2 fwidth(vec2 p) {
        return vec2(fwidth(p.x), fwidth(p.y));
}

// See http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html.
Float approx_log2(Float x) {
    // e - 127 is a fair approximation of log2(x) in its own right...
    Float e = cast(bit_cast<U32>(x)) * (1.0f / (1<<23));

    // ... but using the mantissa to refine its error is _much_ better.
    Float m = bit_cast<Float>((bit_cast<U32>(x) & 0x007fffff) | 0x3f000000);
    return e
         - 124.225514990f
         -   1.498030302f * m
         -   1.725879990f / (0.3520887068f + m);
}
Float approx_pow2(Float x) {
    Float f = fract(x);
    return bit_cast<Float>(round(1.0f * (1<<23),
                             x + 121.274057500f
                               -   1.490129070f * f
                               +  27.728023300f / (4.84252568f - f)));
}

// From skia
Float pow(Float x, Float y) {
    return if_then_else((x == 0)|(x == 1), x
                                         , approx_pow2(approx_log2(x) * y));
}


struct ivec2 {
        ivec2() { vec2(0); }
        ivec2(I32 a): x(a), y(a) {}
        ivec2(I32 x, I32 y): x(x), y(y) {}
        ivec2(U32 x, U32 y): x(__builtin_convertvector(x, I32)), y(__builtin_convertvector(y)) {}
        I32 x;
        I32 y;

        ivec2 operator*=(I32 a) {
                x *= a;
                y *= a;
                return *this;
        }

        friend ivec2 operator*(ivec2 a, I32 b) {
                return ivec2(a.x*b, a.y*b);
        }
};


struct vec3 {
        vec3() { vec3(0); }
        vec3(Float a): x(a), y(a), z(a) {}
        vec3(Float x, Float y, Float z): x(x), y(y), z(z)  {}
        vec3(vec2 a, Float z): x(a.x), y(a.y), z(z)  {}
        Float x;
        Float y;
        Float z;
        friend vec3 operator*(vec3 a, Float b) {
                return vec3(a.x*b, a.y*b, a.z*b);
        }
        friend vec3 operator*(vec3 a, vec3 b) {
                return vec3(a.x*b.x, a.y*b.y, a.z*b.z);
        }

        friend vec3 operator/(vec3 a, Float b) {
                return vec3(a.x/b, a.y/b, a.z/b);
        }


        friend vec3 operator-(vec3 a, Float b) {
                return vec3(a.x-b, a.y-b, a.z-b);
        }
        friend vec3 operator-(vec3 a, vec3 b) {
                return vec3(a.x-b.x, a.y-b.y, a.z-b.z);
        }
        friend vec3 operator+(vec3 a, Float b) {
                return vec3(a.x+b, a.y+b, a.z+b);
        }
        friend vec3 operator+(vec3 a, vec3 b) {
                return vec3(a.x+b.x, a.y+b.y, a.z+b.z);
        }




};
SI vec3 if_then_else(I32 c, vec3 t, vec3 e) {
    return vec3(if_then_else(c, t.x, e.x),
                if_then_else(c, t.y, e.y),
                if_then_else(c, t.z, e.z));
}

SI vec3 clamp(vec3 a, vec3 minVal, vec3 maxVal) {
    return vec3(clamp(a.x, minVal.x, maxVal.x),
                clamp(a.y, minVal.y, maxVal.y),
                clamp(a.z, minVal.z, maxVal.z));
}

vec3 pow(vec3 x, vec3 y) {
    return vec3(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z));
}


U32 uint(I32 x) {
        return __builtin_convertvector(x,   U32);
}

struct vec3_ref {
        vec3_ref(Float &x, Float &y, Float &z) : x(x), y(y), z(z) {
        }
        Float &x;
        Float &y;
        Float &z;
        vec3_ref& operator=(const vec3 &a) {
                x = a.x;
                y = a.y;
                z = a.z;
                return *this;
        }


};

struct vec4 {
        vec4() { vec4(0); }
        vec4(Float a): x(a), y(a), z(a), w(a) {}
        Float& select(XYZW c) {
                switch (c) {
                    case X: return x;
                    case Y: return y;
                    case Z: return z;
                    case W: return w;
                }
        }
        vec3 sel(XYZW c1, XYZW c2, XYZW c3) {
                return vec3(select(c1), select(c2), select(c3));
        }
        vec3_ref lsel(XYZW c1, XYZW c2, XYZW c3) {
                return vec3_ref(select(c1), select(c2), select(c3));
        }

        Float& operator[](int index) {
                switch (index) {
                        case 0: return x;
                        case 1: return y;
                        case 2: return z;
                        case 3: return w;
                }
        }

        // glsl supports non-const indexing of vecs.
        // hlsl doesn't. The code it generates is probably not wonderful.
        Float operator[](I32 index) {
                float sel_x;
                switch (index.x) {
                        case 0: sel_x = x.x; break;
                        case 1: sel_x = y.x; break;
                        case 2: sel_x = z.x; break;
                        case 3: sel_x = w.x; break;
                }
                float sel_y;
                switch (index.y) {
                        case 0: sel_y = x.y; break;
                        case 1: sel_y = y.y; break;
                        case 2: sel_y = z.y; break;
                        case 3: sel_y = w.y; break;
                }
                float sel_z;
                switch (index.z) {
                        case 0: sel_z = x.z; break;
                        case 1: sel_z = y.z; break;
                        case 2: sel_z = z.z; break;
                        case 3: sel_z = w.z; break;
                }
                float sel_w;
                switch (index.w) {
                        case 0: sel_w = x.w; break;
                        case 1: sel_w = y.w; break;
                        case 2: sel_w = z.w; break;
                        case 3: sel_w = w.w; break;
                }
                Float(sel_x, sel_y, sel_z, sel_w);
        }

        Float x;
        Float y;
        Float z;
        Float w;
};

struct sampler2D {
};

}
