/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Yichao Cheng
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


/**
 * bitmap class declaration, GPU supported
 *
 * Authors:
 *      Yichao Cheng (onesuperclark@gmail.com)
 *      Ye Li (mailly1994@gmail.com)
 *
 * Created on 2014-11-29
 * Last modified on 2014-11-30
 */

#ifndef BITMAP_H
#define BITMAP_H

#include <algorithm>

#include "common.h"

class Bitmap : public Managed {
private:
    Word *words;
    int numWords;

public:
    Bitmap() : words(NULL), numWords(0) {}

    /**
     * Allocate a bitmap initalize with zeros on unified memory
     *
     * @param  numBits [number of bits in the bitmap]
     */
    explicit Bitmap(int numBits) {
        numWords = ((numBits - 1) >> 6) + 1;
        cudaMallocManaged(&words, numWords * sizeof(Word));
        memset(words, 0, numWords * sizeof(Word));
    }
    ~Bitmap() {
        cudaFree(words);
    }

    /**
     * Sets the bit at the specified index to 1. Both host and device side
     *
     * @param index [the bit index]
     */
    inline __host__ __device__
    void set(int index) {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
#ifdef __CUDA_ARCH__
        atomicOr(
            reinterpret_cast<unsigned long long *>(&words[index >> 6]),
            static_cast<unsigned long long>(bitmask) );
#else
        words[index >> 6] |= bitmask;
#endif
    }

    /**
     * Sets the bit at the specified index to 0. Both host and device side
     *
     * @param index [the bit index]
     */
    inline __host__ __device__
    void unset(int index) {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
#ifdef __CUDA_ARCH__
        atomicAnd(
            reinterpret_cast<unsigned long long *>(&words[index >> 6]),
            static_cast<unsigned long long>(~bitmask) );
#else
        words[index >> 6] &= ~bitmask;
#endif
    }

    /**
     * Get the bit with the specified index.
     *
     * @param index The bit index
     * @return  [True if the bit is currently set]
     */
    inline __host__ __device__
    bool get(int index) const {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
        return (words[index >> 6] & bitmask) != 0;
    }

    /**
     * Get the capacity (number of bits) contained in this bitmap
     * @return  [the capacity]
     */
    inline __host__ __device__
    int capacity(void) const {
        return numWords << 6;
    }

    /**
     * Deep copy a bitmap to myself
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
    Bitmap &operator= (const Bitmap &other) {
        Word *temp = new Word[other.numWords];
        memcpy(temp, other.words, other.numWords * sizeof(Word));
        if (words != NULL) {
            delete[] words;
        }
        words = temp;
        numWords = other.numWords;
        return * this;
    }

    /**
     * Compute the bit-wise AND of two bitmaps and return it as result.
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
    Bitmap operator& (const Bitmap &other) {
        int numBitsBigger   = std::max(capacity(), other.capacity());
        int numWordsSmaller = std::min(numWords, other.numWords);
        Bitmap newBitmap(numBitsBigger);
        for (int i = 0; i < numWordsSmaller; i++) {
            newBitmap.words[i] = words[i] & other.words[i];
        }
        return newBitmap;
    }

    /**
     * Compute the bit-wise OR of two bitmaps and return it as result.
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
    Bitmap operator| (const Bitmap &other) {
        int numBitsBigger   = std::max(capacity(), other.capacity());
        int numWordsSmaller = std::min(numWords, other.numWords);
        Bitmap newBitmap(numBitsBigger);
        for (int i = 0; i < numWordsSmaller; i++) {
            newBitmap.words[i] = words[i] | other.words[i];
        }
        // The bigger one lasts here and copys the rest of itself
        for (int i = numWordsSmaller; i < numWords; i++) {
            newBitmap.words[i] = words[i];
        }
        for (int i = numWordsSmaller; i < other.numWords; i++) {
            newBitmap.words[i] = other.words[i];
        }
        return newBitmap;
    }

    /**
     * Compute the bit-wise XOR of two bitmaps and return it as result.
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
    Bitmap operator^ (const Bitmap &other) {
        int numBitsBigger   = std::max(capacity(), other.capacity());
        int numWordsSmaller = std::min(numWords, other.numWords);
        Bitmap newBitmap(numBitsBigger);
        for (int i = 0; i < numWordsSmaller; i++) {
            newBitmap.words[i] = words[i] ^ other.words[i];
        }
        // The bigger one lasts here and copys the rest of itself
        for (int i = numWordsSmaller; i < numWords; i++) {
            newBitmap.words[i] = words[i];
        }
        for (int i = numWordsSmaller; i < other.numWords; i++) {
            newBitmap.words[i] = other.words[i];
        }
        return newBitmap;
    }
};

#endif  // BITMAP_H
