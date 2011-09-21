#ifndef MSVCEXPORTS_H
#define MSVCEXPORTS_H

#ifdef MSVC_VER
#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif
#else
#define DLL_API
#endif

#endif // MSVCEXPORTS_H
