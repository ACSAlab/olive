/**
 * Logger
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */


#include <stdio.h>

/**
 * Any class requires logging can inheritate this class 
 */
class Logging {
 private:
    FILE * logFile;

 public:
    /**
     * Change the logging redirection. By default, the log content is printed on
     * the screen.
     * @param file log file
     */
    void log(FILE * file) {
        logFile = file
    }

    void logInfo(...) {
        fprintf(logFile, "[Info] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

    void logWarning(...) {
        fprintf(logFile, "[Warning] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

    void logDebug(...) {
        fprintf(logFile, "[Debug] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, ": %s (%d)", __FILE__, __LINE__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

    void logError(...) {
        fprintf(logFile, "[Error] ");
        fprintf(logFile, __VA_ARGS__);
        fprintf(logFile, ": %s (%d)", __FILE__, __LINE__);
        fprintf(logFile, "\n");
        fflush(logFile);
    }

    Loggging() : logFile(stdout) {}
}




