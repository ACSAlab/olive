/**
 * A simple bit set implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */

#ifndef BITMAP_H
#define BITMAP_H

#include <algorithm>

#include "common.h"
#include "utils.h"

/**
 * A simple bitmap implementation. No bound checking so it is fast.
 */
class Bitmap {
 private:
    Word * words;
    int numWords;

 public:
    Bitmap(void) {
        numWords = 0;
        words = NULL;
    }

    /**
     * Allocate a bitmap initalize with zeros on the heap.
     * 
     * @param numBits number of bits in the bitmap
     */
     explicit Bitmap(int numBits) {
        numWords = ((numBits - 1) >> 6) + 1;
        words = new Word[numWords]();
    }

    ~Bitmap(void) {
        delete[] words;
    }

    /** Get the capacity (number of bits) contained in this bitmap */
    int capacity(void) const {
        return numWords << 6;
    }

    /**
    * Sets the bit at the specified index to 1.
    * 
    * @param index the bit index
    */
    void set(int index) {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
        words[index >> 6] |= bitmask;
    }

    /**
    * Sets the bit at the specified index to 0.
    * 
    * @param index the bit index
    */
    void unset(int index) {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
        words[index >> 6] &= ~bitmask;
    }

    /**
     * Get the bit with the specified index.
     *
     * @param index The bit index
     * @return      True if the bit is currently set
     */
    bool get(int index) const {
        Word bitmask = static_cast<Word>(1) << (index & 0x3f);
        return (words[index >> 6] & bitmask) != 0;
    }

    /** 
     * Deep copy a bitmap to myself
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
     Bitmap & operator= (const Bitmap & other) {
        Word * temp = new Word[other.numWords];
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
     Bitmap operator& (const Bitmap & other) {
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
     Bitmap operator| (const Bitmap & other) {
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
