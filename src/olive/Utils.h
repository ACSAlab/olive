/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */

#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <sys/time.h>

namespace util {

/**
 * Checks if the string is a numeric number
 * @param  str String to check
 * @return     True If the string represents a numeric number
 */
bool isNumeric(const char * str) {
  assert(str);
  while ((* str) != '\0') {
    if (!isdigit(* str)) {
        return false;
    } else {
        str++;
    }
  }
  return true;
}

/**
 * Get the hash code of any given number very quickly
 * @param  a The number to hash
 * @return   The hash code
 */
size_t hashCode(size_t a) {
      a ^= (a << 13);
      a ^= (a >>> 17);
      a ^= (a << 5);
      return a;
}

/** Get current time in milliseconds */
double currentTimeMillis(void) {
    timeval t;
    gettimeofday(&t, NULL);
    double millis = static_cast<double> (t.tv_sec * 1000.0);
    millis += static_cast<double >(t.tv_usec / 1000.0);
    return millis;
}

/** Get current time in milliseconds */
double currentTimeSeconds(void) {
    timeval t;
    gettimeofday(&t, NULL);
    double seconds = static_cast<double> (t.tv_sec);
    seconds += static_cast<double>(t.tv_usec / 1000000.0);
    return seconds;
}

}  // namespace util

#endif  // UTILS_H
