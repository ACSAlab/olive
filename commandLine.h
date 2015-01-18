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
 * Commandline Parser
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-01-16
 * Last Modified: 2015-01-16
 */

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

class CommandLine {
private:
    int argc;
    char **argv;
    std::string example;

public:
    CommandLine(int c, char **v, std::string s): argc(c), argv(v), example(s) {}

    void badArgument() {
        std::cout << "usage: " << argv[0] << " " << example << std::endl;
        abort();
    }

    // get an argument,  0-> argv[2], 1-> argv[3]
    char *getArgument(int i) {
        if (argc < i + 2) badArgument();
        return argv[i+1];
    }

    // Find an option in the comand line string
    bool getOption(std::string option) {
        for (int i = 1; i < argc; i++)
            if ((std::string) argv[i] == option) return true;
        return false;
    }

    /**
     * prog ... -option value ...
     */
    int getOptionIntValue(std::string option, int defaultValue) {
        for (int i = 1; i < argc - 1; i++)
            if ((std::string) argv[i] == option) {
                int r = atoi(argv[i + 1]);
                return r;
            }
        return defaultValue;
    }

    long getOptionLongValue(std::string option, long defaultValue) {
        for (int i = 1; i < argc - 1; i++)
            if ((std::string) argv[i] == option) {
                long r = atol(argv[i + 1]);
                return r;
            }
        return defaultValue;
    }

    double getOptionDoubleValue(std::string option, double defaultValue) {
        for (int i = 1; i < argc - 1; i++)
            if ((std::string) argv[i] == option) {
                double val;
                if (sscanf(argv[i + 1], "%lf",  &val) == EOF) {
                    badArgument();
                }
                return val;
            }
        return defaultValue;
    }
};


#endif  // COMMAND_LINE_H