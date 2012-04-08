#ifndef CONFIG_H
#define CONFIG_H

/*
 * Configuration and environment-specific features go here
 */

//MSVC doesn't have std::isnan
#ifdef _MSC_VER
#include <cfloat>
namespace std{
template<class T> int isnan(T x) { return x!=x; }
}
#endif

//Convert eigen asserts to exceptions - much safer
#include <stdexcept>
#define eigen_assert(X) if(!(X)) {throw std::runtime_error(#X);}

//Workaround for Eigen alignment bug
#if defined (__MINGW32__)
#define ALIGN_EIGEN_FUNCTION __attribute__((force_align_arg_pointer))
#else
#define ALIGN_EIGEN_FUNCTION
#endif

#endif // CONFIG_H
