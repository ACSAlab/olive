/**
 * Logger
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */

#ifndef LOGGING_H
#define LOGGING_H

#include "common.h"

/**
 * Any class requires logging can inherit this class 
 */
class Logging {
 private:
	 static FILE * logFile;

 public:
    /**
     * Change the logging redirection.
     * @param file New log file
     */
	 static void setLogFile(FILE * file) {
		logFile = file;
    }

    static void logInfo(...) {
        fprintf(logFile, "[Info] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

	static void logWarning(...) {
        fprintf(logFile, "[Warning] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

	static void logDebug(...) {
        fprintf(logFile, "[Debug] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, ": %s (%d)", __FILE__, __LINE__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

	static void logError(...) {
        fprintf(logFile, "[Error] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, ": %s (%d)", __FILE__, __LINE__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

    /** By default, logs are outputed to stdout */
};

#endif  // LOGGING_H


