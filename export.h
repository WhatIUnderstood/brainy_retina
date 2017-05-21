#ifndef EXPORT_H
#define EXPORT_H

#if BUILDING_LIBTEMPU && HAVE_VISIBILITY
#define LIBTEMPU_DLL_EXPORTED __attribute__((__visibility__("default")))
#else
#define LIBTEMPU_DLL_EXPORTED
#endif

#endif // EXPORT_H
