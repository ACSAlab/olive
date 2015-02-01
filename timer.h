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
 * Timer.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

/** Get current time in milliseconds */
double getTimeMillis() {
    timeval t;
    gettimeofday(&t, NULL);
    double millis = static_cast<double> (t.tv_sec * 1000.0);
    millis += static_cast<double >(t.tv_usec / 1000.0);
    return millis;
}

/** Get current time in seconds */
double getTimeSeconds() {
    timeval t;
    gettimeofday(&t, NULL);
    double seconds = static_cast<double> (t.tv_sec);
    seconds += static_cast<double>(t.tv_usec / 1000000.0);
    return seconds;
}

/** The Stopwatch returns the elapsed time to last time when it stops */
class Stopwatch {
private:
    double timer_;

public:
    void start() {
        timer_ = getTimeMillis();
    }

    double getElapsedMillis() {
        double now = getTimeMillis();
        double elapsed = now - timer_;
        timer_ = now;
        return elapsed;
    }
};


#endif  // TIMER_H