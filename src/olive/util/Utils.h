/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#pragma once

#include <time.h>
#include <ctype.h>
#include <string.h>


/**
 * Checks if the string is a numeric number
 * @param  str   The string to check
 * @return true  If the string represents a numeric number
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
    seconds += static_cast<double>(t.tv_usec/1000000.0);
    return seconds;
}
