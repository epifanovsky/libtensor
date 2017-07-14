#ifndef LIBUTIL_BACKTRACE_H
#define LIBUTIL_BACKTRACE_H

#include <cstdlib>

namespace libutil {


/** \brief Generates a stack trace and stores it

    \ingroup libutil_exceptions
 **/
class backtrace {
public:
    enum {
        k_buflen = 16384,
        k_maxframes = 256
    };

private:
    char m_buf[k_buflen];
    char *m_frames[k_maxframes];
    size_t m_nframes;

public:
    /** \brief Constructs the stack trace by loading it from the OS
     **/
    backtrace();

    /** \brief Copy constructor
     **/
    backtrace(const backtrace &bt);

    /** \brief Returns the number of stack frames
     **/
    size_t get_nframes() const {
        return m_nframes;
    }

    /** \brief Returns the contents of a stack frame, frame number shall not
            exceed the value returned by get_nframes()
     **/
    const char *get_frame(size_t i) const {
        return m_frames[i];
    }

};


} // namespace libutil

#endif // LIBUTIL_BACKTRACE_H
