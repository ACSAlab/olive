/**
 * A simple bitset implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */

#pragma once

#include <cinttypes>
#include <cstdlib>
#include <cstring>


/** A word contains 64 bits */
typedef uint64_t Word;


#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)


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
     * Allocate a bitmap on the heap. Note that the inital values are not zero.
     * 
     * @param numBits number of bits in the bitmap
     */
    explicit Bitmap(int numBits) {
        numWords = ((numBits - 1) >> 6) + 1;  // Div by 64 conservatively
        words = new Word[numWords];
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
        Word bitmask = 1 << (index & 0x3f);     // Mod 64 and shift
        words[index >> 6] |= bitmask;           // Div by 64 and mask
    }

    void unset(int index) {
        Word bitmask = 1 << (index & 0x3f);     // Mod 64 and shift
        words[index >> 6] &= ~bitmask;          // Div by 64 and mask
    }

    /**
    * Get the bit with the specified index.
    *
    * @param index The bit index
    * @return      True if the bit is currently set
    */
    bool get(int index) const {
        Word bitmask = 1 << (index & 0x3f);
        return (words[index >> 6] & bitmask) != 0;
    }

    /** 
     * Deep copy a bitmap to myself
     *
     * @param other The other bitmap
     * @return      Result bitmap
     */
    Bitmap & operator= (const Bitmap & other) {
        // Copy other to a temp buffer first
        Word * temp = new Word[other.numWords];
        memcpy(temp, other.words, other.numWords * sizeof(Word));
        if (words != NULL) {
            delete[] words;             // Delete the old data
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
        int numBitsBigger   = MAX(capacity(), other.capacity());
        int numWordsSmaller = MIN(numWords, other.numWords);
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
        int numBitsBigger   = MAX(capacity(), other.capacity());
        int numWordsSmaller = MIN(numWords, other.numWords);
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
        int numBitsBigger   = MAX(capacity(), other.capacity());
        int numWordsSmaller = MIN(numWords, other.numWords);
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

    ~Bitmap(void) {
        delete[] words;
        words = NULL;
        numWords = 0;
    }
};

