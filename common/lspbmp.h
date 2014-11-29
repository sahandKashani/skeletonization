#ifndef INCLUDED_LSPBMP_H
#define INCLUDED_LSPBMP_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct Bitmap {
    int width;
    int height;
    int depth;
    unsigned char data[0];
} Bitmap;

Bitmap *loadBitmap(const char *fname);
int saveBitmap(const char *fname, Bitmap *bmp);
Bitmap *createBitmap(int w, int h, int d);

#ifdef __cplusplus
}
#endif

#endif
